# Protótipo Etapa 2 — ByteTrack + OSNet

Aplicação local para vídeo de câmera única. Preserva a Etapa 1 e adiciona um módulo experimental de re-identificação visual.

## Funcionalidades

- Modo baseline: YOLO + ByteTrack.
- Modo Re-ID: YOLO + ByteTrack + OSNet (`osnet_x0_25`).
- IDs locais do ByteTrack separados dos IDs globais do protótipo.
- Galeria temporária de embeddings por identidade global.
- Comparação por similaridade de cosseno.
- Critério de margem entre o melhor e o segundo melhor candidato.
- Atualização gradual do protótipo visual da identidade.
- Vídeo anotado, CSV de detecções/decisões, CSV de eventos e relatório JSON.

## Instalação (Windows)

Recomendado: Python 3.10 ou 3.11. Crie um ambiente virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instale o PyTorch adequado ao seu computador seguindo o seletor oficial:
https://pytorch.org/get-started/locally/

Para instalação CPU, uma opção comum é:

```powershell
pip install torch torchvision
```

Depois:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Se a instalação de `torchreid` falhar no seu ambiente, consulte a documentação do projeto Torchreid/Deep-Person-ReID e instale-o no mesmo ambiente Python. A aplicação informa explicitamente quando o módulo não está disponível.

## Uso

1. Execute `streamlit run app.py`.
2. Escolha `ByteTrack (baseline)` ou `ByteTrack + OSNet (Re-ID)`.
3. Envie o vídeo.
4. Clique em `Analisar vídeo`.
5. Baixe o vídeo anotado, CSVs e relatório.

Na primeira execução, os pesos do YOLO e os pesos pré-treinados do OSNet podem ser baixados. A análise do vídeo é local.

## Como interpretar

- `bytetrack_local_id`: ID produzido pelo rastreador local.
- `global_id`: ID atribuído pelo protótipo; no modo Re-ID pode ser recuperado após uma interrupção.
- `reid_cosine_similarity`: similaridade entre o embedding atual e o protótipo visual candidato.
- `second_best_similarity`: similaridade do segundo candidato.
- `similarity_margin`: diferença entre o melhor e o segundo candidato.
- `absence_candidate_frames`: quadros desde a última observação do candidato.
- `decision`: continuidade, aceitação de Re-ID ou criação de novo ID.

## Limitações científicas importantes

1. A similaridade de aparência não prova que duas detecções pertencem à mesma pessoa. Erros de associação são possíveis, especialmente com roupas semelhantes, baixa resolução, mudanças de pose, iluminação ou oclusão.
2. O ID global é uma hipótese computacional do protótipo, não uma identidade civil ou biométrica confirmada.
3. A galeria é mantida apenas durante o processamento de um vídeo; não existe persistência entre execuções.
4. Este programa continua sendo de câmera única. Não resolve associação entre câmeras não sobrepostas.
5. IDF1, MOTA, precisão de Re-ID e taxa real de recuperação exigem ground truth anotado. O relatório fornece contagens e métricas operacionais, não acurácia validada.
6. O registro de similaridade e margem é uma evidência operacional auditável, não uma explicação causal completa do modelo OSNet.

## Protocolo experimental sugerido

Para comparação justa, execute o mesmo vídeo em ambos os modos com os mesmos parâmetros de detector e ByteTrack. Registre:
- número de IDs distintos;
- número de fragmentações/reaparecimentos;
- tentativas e aceitações de Re-ID;
- FPS e tempo total;
- erros de associação observados manualmente.

Depois, anote manualmente uma amostra de trajetórias para calcular métricas baseadas em ground truth. Não conclua que menos IDs significa melhor desempenho: uma associação incorreta também pode reduzir a contagem de IDs.
