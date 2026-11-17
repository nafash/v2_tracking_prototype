import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(page_title="Rastreamento + Re-ID", page_icon="🎥", layout="wide")
st.title("Rastreamento de Pessoas — Etapa 2")
st.caption("ByteTrack + OSNet • Re-identificação em vídeo de câmera única")

st.info(
    "O sistema processa um vídeo de uma câmera. O OSNet compara a aparência de pessoas "
    "que voltam a aparecer após uma interrupção do rastreamento. Isso ainda não é Re-ID "
    "entre câmeras diferentes."
)

with st.sidebar:
    st.header("Detecção e rastreamento")
    detector_model = st.selectbox("Modelo YOLO", ["yolo11n.pt", "yolo11s.pt"], index=0)
    confidence = st.slider("Confiança mínima", 0.10, 0.90, 0.35, 0.05)
    track_iou = st.slider("IoU do ByteTrack", 0.10, 0.90, 0.50, 0.05)
    track_buffer = st.slider("Buffer ByteTrack (quadros)", 10, 180, 60, 10)
    detector_device = st.selectbox(
        "Dispositivo do detector", ["cpu", "0"], index=0,
        help="Use '0' somente se o PyTorch/Ultralytics reconhecer uma GPU CUDA compatível."
    )

    st.divider()
    st.header("Re-identificação")
    mode = st.radio("Modo experimental", ["ByteTrack (baseline)", "ByteTrack + OSNet (Re-ID)"])
    reid_threshold = st.slider("Similaridade mínima (cosseno)", 0.30, 0.95, 0.72, 0.01)
    reid_margin = st.slider("Margem sobre o segundo candidato", 0.00, 0.30, 0.05, 0.01)
    min_gap = st.slider("Ausência mínima para tentar Re-ID (quadros)", 1, 120, 15, 1)
    max_gallery_age = st.slider("Validade da galeria (quadros)", 60, 3000, 900, 60)
    st.caption("Ajuste esses limiares experimentalmente. Não existe um valor universal.")

uploaded = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov", "mkv"])

def build_osnet(device):
    """Carrega OSNet pelo pacote torchreid. Falha explicitamente se não estiver instalado."""
    try:
        import torchreid
    except Exception as exc:
        raise RuntimeError(
            "O pacote torchreid não está disponível. Instale as dependências da Etapa 2 "
            "conforme o README e reinicie a aplicação."
        ) from exc

    model = torchreid.models.build_model(
        name="osnet_x0_25",
        num_classes=1000,
        loss="softmax",
        pretrained=True
    )
    model.eval()
    model.to(device)
    return model

def crop_tensor(frame, xyxy):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

@torch.inference_mode()
def extract_embeddings(model, frame, boxes, device):
    tensors = [crop_tensor(frame, b) for b in boxes]
    valid_indices = [i for i, t in enumerate(tensors) if t is not None]
    if not valid_indices:
        return {}
    batch = torch.stack([tensors[i] for i in valid_indices]).to(device)
    features = model(batch)
    if isinstance(features, (tuple, list)):
        features = features[0]
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    features = features.detach().cpu().numpy()
    return {idx: features[pos].astype(np.float32) for pos, idx in enumerate(valid_indices)}

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def process_video(input_path, output_path, detections_path, events_path, report_path):
    use_reid = mode == "ByteTrack + OSNet (Re-ID)"
    reid_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reid_model = build_osnet(reid_device) if use_reid else None
    detector = YOLO(detector_model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo enviado.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Não foi possível criar o vídeo de saída.")

    tracker_yaml = Path(tempfile.gettempdir()) / "etapa2_bytetrack.yaml"
    tracker_yaml.write_text(
        f"""tracker_type: bytetrack
track_high_thresh: 0.45
track_low_thresh: 0.10
new_track_thresh: 0.50
track_buffer: {track_buffer}
match_thresh: {track_iou}
fuse_score: True
""", encoding="utf-8"
    )

    # Galeria de protótipos visuais por identidade global.
    gallery = {}             # global_id -> {embedding, last_seen, hits}
    local_to_global = {}     # ID temporário ByteTrack -> ID global
    global_last_seen = {}
    next_global_id = 1
    detection_rows, event_rows = [], []
    frame_index = 0
    total_detections = 0
    reid_attempts = 0
    reid_matches = 0
    started = time.time()
    progress = st.progress(0.0, text="Iniciando...")
    status = st.empty()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            result = detector.track(
                frame, persist=True, tracker=str(tracker_yaml), classes=[0],
                conf=confidence, iou=track_iou, device=detector_device, verbose=False
            )[0]

            boxes_xyxy, local_ids, det_confs = [], [], []
            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                local_ids = result.boxes.id.int().cpu().tolist()
                det_confs = result.boxes.conf.cpu().numpy().tolist()

            embeddings = extract_embeddings(reid_model, frame, boxes_xyxy, reid_device) if use_reid and len(boxes_xyxy) else {}
            active_global_ids = set()
            annotations = []

            for idx, (box, local_id) in enumerate(zip(boxes_xyxy, local_ids)):
                if local_id is None:
                    continue
                local_id = int(local_id)
                emb = embeddings.get(idx)
                conf = float(det_confs[idx]) if idx < len(det_confs) else None
                decision = "continuidade_bytetrack"
                best_score, second_score, gap = None, None, None
                matched_from = None

                if local_id in local_to_global:
                    global_id = local_to_global[local_id]
                elif use_reid and emb is not None:
                    # Só considera identidades não observadas recentemente, evitando
                    # atribuir o mesmo ID global a duas pessoas simultâneas.
                    candidates = []
                    for gid, item in gallery.items():
                        absence = frame_index - item["last_seen"]
                        if gid in active_global_ids:
                            continue
                        if min_gap <= absence <= max_gallery_age:
                            candidates.append((cosine(emb, item["embedding"]), gid, absence))
                    candidates.sort(reverse=True, key=lambda x: x[0])
                    reid_attempts += 1
                    if candidates:
                        best_score, candidate_id, gap = candidates[0]
                        second_score = candidates[1][0] if len(candidates) > 1 else None
                        margin_ok = second_score is None or (best_score - second_score) >= reid_margin
                        if best_score >= reid_threshold and margin_ok:
                            global_id = candidate_id
                            decision = "reid_correspondencia_aceita"
                            matched_from = candidate_id
                            reid_matches += 1
                        else:
                            global_id = next_global_id
                            next_global_id += 1
                            decision = "novo_id_similaridade_insuficiente"
                    else:
                        global_id = next_global_id
                        next_global_id += 1
                        decision = "novo_id_sem_candidato_elegivel"
                    local_to_global[local_id] = global_id
                else:
                    if use_reid:
                        # Sem embedding válido não tentamos comparar aparências.
                        # Ainda assim, atribuímos um ID global novo sem colidir com
                        # o contador de identidades criado pelo módulo de Re-ID.
                        global_id = next_global_id
                        next_global_id += 1
                        local_to_global[local_id] = global_id
                        decision = "novo_id_sem_embedding_crop_invalido"
                    else:
                        # No baseline, preserva o ID local produzido pelo ByteTrack.
                        global_id = local_id
                        local_to_global[local_id] = global_id

                active_global_ids.add(global_id)
                previous_seen = global_last_seen.get(global_id)
                gap_since_seen = frame_index - previous_seen - 1 if previous_seen is not None else 0
                global_last_seen[global_id] = frame_index

                if use_reid and emb is not None:
                    if global_id in gallery:
                        old = gallery[global_id]["embedding"]
                        # Atualização lenta do protótipo para reduzir ruído entre quadros.
                        updated = 0.90 * old + 0.10 * emb
                        updated /= (np.linalg.norm(updated) + 1e-12)
                        gallery[global_id]["embedding"] = updated.astype(np.float32)
                        gallery[global_id]["hits"] += 1
                    else:
                        gallery[global_id] = {"embedding": emb, "last_seen": frame_index, "hits": 1}
                    gallery[global_id]["last_seen"] = frame_index

                x1, y1, x2, y2 = [int(v) for v in box]
                total_detections += 1
                detection_rows.append({
                    "frame": frame_index,
                    "time_seconds": round((frame_index - 1) / fps, 3),
                    "global_id": global_id,
                    "bytetrack_local_id": local_id,
                    "detector_confidence": round(conf, 4) if conf is not None else None,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "gap_since_global_id_seen_frames": gap_since_seen,
                    "decision": decision,
                    "reid_cosine_similarity": round(best_score, 4) if best_score is not None else None,
                    "second_best_similarity": round(second_score, 4) if second_score is not None else None,
                    "similarity_margin": round(best_score - second_score, 4) if best_score is not None and second_score is not None else None,
                    "absence_candidate_frames": gap,
                    "evidence": "ByteTrack local ID" if not use_reid else "ByteTrack + embedding OSNet + similaridade de cosseno"
                })
                event_rows.append({
                    "frame": frame_index, "global_id": global_id, "local_id": local_id,
                    "event": decision, "best_similarity": best_score,
                    "second_best_similarity": second_score, "absence_frames": gap
                })

                color = (40, 210, 80) if decision in ("continuidade_bytetrack", "reid_correspondencia_aceita") else (0, 190, 255)
                label = f"ID {global_id} | local {local_id}"
                if best_score is not None:
                    label += f" | sim {best_score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                top = max(0, y1 - 25)
                cv2.rectangle(frame, (x1, top), (min(width, x1 + 250), y1), color, -1)
                cv2.putText(frame, label, (x1 + 4, max(16, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

            writer.write(frame)
            if total_frames:
                progress.progress(min(frame_index / total_frames, 1.0),
                                  text=f"Processando quadro {frame_index}/{total_frames}")
            if frame_index % 30 == 0:
                elapsed = max(time.time() - started, 1e-6)
                status.write(f"Quadros: {frame_index} • FPS processamento: {frame_index / elapsed:.2f}")

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - started
    df = pd.DataFrame(detection_rows)
    events = pd.DataFrame(event_rows)
    df.to_csv(detections_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")

    report = {
        "mode": mode,
        "frames_processed": frame_index,
        "source_fps": fps,
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frame_index / max(elapsed, 1e-6), 3),
        "detections_with_track_id": total_detections,
        "unique_global_ids": int(df["global_id"].nunique()) if not df.empty else 0,
        "reid_attempts": reid_attempts,
        "reid_accepted_matches": reid_matches,
        "reid_threshold": reid_threshold if use_reid else None,
        "reid_margin": reid_margin if use_reid else None,
        "reid_device": str(reid_device) if use_reid else None,
        "warning": (
            "Resultados exploratórios. Similaridade visual não comprova identidade. "
            "IDF1/MOTA e taxa real de recuperação exigem ground truth anotado."
        )
    }
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.progress(1.0, text="Concluído.")
    return report, df, events

if uploaded:
    st.video(uploaded)
    if st.button("Analisar vídeo", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as workdir:
            input_path = str(Path(workdir) / uploaded.name)
            output_path = str(Path(workdir) / "video_etapa2_anotado.mp4")
            detections_path = str(Path(workdir) / "deteccoes.csv")
            events_path = str(Path(workdir) / "eventos_reid.csv")
            report_path = str(Path(workdir) / "relatorio_etapa2.json")
            Path(input_path).write_bytes(uploaded.getvalue())
            try:
                with st.spinner("Carregando modelos e processando vídeo..."):
                    report, detections, events = process_video(
                        input_path, output_path, detections_path, events_path, report_path
                    )
                st.success("Processamento concluído.")
                a, b, c, d = st.columns(4)
                a.metric("Quadros", report["frames_processed"])
                b.metric("IDs globais", report["unique_global_ids"])
                c.metric("FPS processamento", report["processing_fps"])
                d.metric("Re-ID aceitas", report["reid_accepted_matches"])
                if mode == "ByteTrack + OSNet (Re-ID)":
                    st.caption(f"Tentativas de Re-ID: {report['reid_attempts']} • Dispositivo OSNet: {report['reid_device']}")
                st.subheader("Vídeo anotado")
                video_bytes = Path(output_path).read_bytes()
                st.video(video_bytes)
                st.download_button("Baixar vídeo anotado", video_bytes,
                                   "video_etapa2_anotado.mp4", "video/mp4")
                st.subheader("Detecções e decisões")
                st.dataframe(detections.head(1000), use_container_width=True)
                st.download_button("Baixar detecções (CSV)", Path(detections_path).read_bytes(),
                                   "deteccoes.csv", "text/csv")
                st.subheader("Eventos de associação")
                st.dataframe(events.head(1000), use_container_width=True)
                st.download_button("Baixar eventos Re-ID (CSV)", Path(events_path).read_bytes(),
                                   "eventos_reid.csv", "text/csv")
                st.download_button("Baixar relatório (JSON)", Path(report_path).read_bytes(),
                                   "relatorio_etapa2.json", "application/json")
                st.warning(report["warning"])
            except Exception as exc:
                st.error(f"Erro no processamento: {exc}")
else:
    st.markdown("### Como usar")
    st.write("Selecione o modo experimental, envie um vídeo e clique em **Analisar vídeo**.")
    st.write("Para comparar os métodos, execute o mesmo vídeo uma vez no modo baseline e outra com OSNet, mantendo os mesmos parâmetros.")
