#!/usr/bin/env bash
#
# Executa os experimentos em container, para rodar num servidor remoto com GPU.
#
# Uso mais comum:
#   ./deploy.sh doctor            # confere pre-requisitos (faca isso primeiro)
#   ./deploy.sh data              # baixa o FINDSum (6,5 GB) e gera a particao
#   ./deploy.sh test_50           # roda o perfil configs/test_50.yaml
#   ./deploy.sh eval_1000         # avaliacao final (~71 h)
#
# Outros comandos:
#   ./deploy.sh build             # so constroi a imagem
#   ./deploy.sh compare <perfil>  # refaz a tabela estatistica
#   ./deploy.sh pack <perfil>     # reempacota o tarball
#   ./deploy.sh shell             # shell dentro do container
#   ./deploy.sh test              # roda a suite de testes no container
#
# ONDE FICA A SAIDA (tudo no host, sobrevive ao container):
#
#   results/<perfil>/    LEVE e VERSIONADO no git. Metricas agregadas, scores
#                        por documento e a tabela estatistica. ~50 KB.
#                        -> e isto que voce publica no git.
#
#   outputs/<perfil>/    COMPLETO e fora do git. Inclui predictions.jsonl com
#                        cada resumo gerado e sua referencia. ~5 MB em 50
#                        documentos, ~100 MB em 1.000.
#
#   dist/<perfil>.tar.gz Tudo de outputs/<perfil> comprimido, para baixar num
#                        arquivo unico com scp.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE="${FINDSUM_IMAGE:-findsum-rag:latest}"
readonly HF_CACHE="${FINDSUM_HF_CACHE:-$HOME/.cache/findsum-hf}"
# Permite `FINDSUM_DOCKER="sudo docker"` quando o usuario nao esta no grupo
# docker, ou `FINDSUM_DOCKER=podman` num servidor sem docker.
readonly DOCKER="${FINDSUM_DOCKER:-docker}"
readonly DATA_DIR="$REPO_ROOT/data"
readonly OUT_DIR="$REPO_ROOT/outputs"
readonly RESULTS_DIR="$REPO_ROOT/results"
readonly DIST_DIR="$REPO_ROOT/dist"
readonly MANIFEST="$DATA_DIR/interim/splits-liquidity.json"

# Artefatos que sao pequenos o bastante para versionar. predictions.jsonl fica
# de fora de proposito: e o arquivo que cresce com o numero de documentos.
readonly LIGHT_ARTIFACTS=("summary.json" "reference_baseline.json" "comparison.txt" "config.yaml")

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[aviso]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- pre-requisitos

check_docker() {
  command -v "${DOCKER%% *}" >/dev/null 2>&1 \
    || die "'${DOCKER%% *}' nao encontrado no PATH"
  $DOCKER info >/dev/null 2>&1 || die "$(cat <<EOF
'$DOCKER info' falhou. Causas usuais:

  1. O daemon nao esta rodando:   sudo systemctl start docker
  2. Seu usuario nao esta no grupo docker. Duas saidas:
       a) usar sudo apenas aqui:  FINDSUM_DOCKER="sudo docker" ./deploy.sh ...
       b) entrar no grupo:        sudo usermod -aG docker \$USER && newgrp docker
          (atencao: o grupo docker equivale a acesso root na maquina)
EOF
)"
}

# Devolve as flags de GPU, ou vazio quando o toolkit nao esta presente.
gpu_flags() {
  if $DOCKER info 2>/dev/null | grep -q 'nvidia'; then
    printf '%s' "--gpus all"
  elif command -v nvidia-container-cli >/dev/null 2>&1; then
    printf '%s' "--gpus all"
  else
    printf '%s' ""
  fi
}

require_gpu() {
  local flags
  flags="$(gpu_flags)"
  [[ -n "$flags" ]] || die "$(cat <<'EOF'
GPU nao disponivel para o docker: o nvidia-container-toolkit nao esta instalado.

A geracao exige GPU -- em CPU cada documento levaria horas em vez de ~45 s.

Para instalar no servidor (Ubuntu/Debian):
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

Depois confirme com: ./deploy.sh doctor
EOF
)"
}

doctor() {
  log "docker"
  command -v "${DOCKER%% *}" >/dev/null 2>&1 && $DOCKER --version || warn "docker ausente"
  $DOCKER info >/dev/null 2>&1 && echo "  daemon respondendo: ok" || warn "daemon nao responde"

  log "GPU"
  if [[ -n "$(gpu_flags)" ]]; then
    echo "  runtime nvidia: ok"
    $DOCKER run --rm --gpus all "$IMAGE" \
      python -c "import torch; print('  torch cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" \
      2>/dev/null || warn "imagem ainda nao construida ou GPU inacessivel no container"
  else
    warn "nvidia-container-toolkit ausente: 'run' vai falhar (a geracao exige GPU)"
  fi
  command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /' \
    || warn "nvidia-smi ausente no host"

  log "imagem"
  $DOCKER image inspect "$IMAGE" >/dev/null 2>&1 \
    && echo "  $IMAGE presente ($($DOCKER image inspect -f '{{.Size}}' "$IMAGE" | numfmt --to=iec))" \
    || warn "$IMAGE nao construida (rode: ./deploy.sh build)"

  log "dados"
  if [[ -d "$DATA_DIR/raw/findsum/text" ]]; then
    echo "  FINDSum: $(du -sh "$DATA_DIR/raw/findsum" 2>/dev/null | cut -f1)"
  else
    warn "FINDSum ausente (rode: ./deploy.sh data)"
  fi
  [[ -f "$MANIFEST" ]] \
    && echo "  particao: presente" \
    || warn "particao ausente (rode: ./deploy.sh data)"

  log "cache de modelos"
  mkdir -p "$HF_CACHE"
  echo "  $HF_CACHE ($(du -sh "$HF_CACHE" 2>/dev/null | cut -f1))"

  log "espaco em disco"
  df -h "$REPO_ROOT" | tail -1 | sed 's/^/  /'
  echo
  echo "necessario: ~7 GB de dataset, ~7 GB de imagem, ~6 GB de pesos do modelo"
}

# ------------------------------------------------------------------------ docker

build() {
  check_docker
  log "construindo $IMAGE (a primeira vez baixa ~3 GB de torch)"
  $DOCKER build -t "$IMAGE" "$REPO_ROOT"
  log "pronto: $($DOCKER image inspect -f '{{.Size}}' "$IMAGE" | numfmt --to=iec)"
}

ensure_image() {
  $DOCKER image inspect "$IMAGE" >/dev/null 2>&1 || build
}

# Roda um comando no container. Usa o UID do host para que os arquivos gravados
# nos volumes nao saiam pertencendo ao root.
run_in_container() {
  local -a gpu=()
  local flags
  flags="$(gpu_flags)"
  [[ -n "$flags" ]] && gpu=(--gpus all)

  mkdir -p "$HF_CACHE" "$OUT_DIR" "$RESULTS_DIR" "$DIST_DIR" "$DATA_DIR"

  $DOCKER run --rm -i \
    "${gpu[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e HF_HOME=/cache/huggingface \
    -v "$DATA_DIR:/app/data" \
    -v "$OUT_DIR:/app/outputs" \
    -v "$RESULTS_DIR:/app/results" \
    -v "$DIST_DIR:/app/dist" \
    -v "$HF_CACHE:/cache/huggingface" \
    -w /app \
    "$IMAGE" "$@"
}

# -------------------------------------------------------------------------- dados

fetch_data() {
  check_docker
  ensure_image
  log "baixando o FINDSum (6,5 GB; pula o que ja existe)"
  run_in_container python scripts/fetch_findsum.py

  if [[ -f "$MANIFEST" ]]; then
    log "particao ja existe, mantendo"
    warn "a particao e congelada de proposito. Refaze-la depois de olhar"
    warn "resultados invalida a avaliacao final. Para forcar, apague $MANIFEST"
  else
    log "gerando a particao dos conjuntos experimentais"
    run_in_container python scripts/build_splits.py
  fi
  run_in_container findsum splits
}

# ---------------------------------------------------------------------- execucao

profile_config() {
  local profile="$1"
  local path="configs/${profile}.yaml"
  [[ -f "$REPO_ROOT/$path" ]] || die "perfil '$profile' nao existe ($path). Disponiveis: $(
    cd "$REPO_ROOT/configs" && ls -1 *.yaml 2>/dev/null | sed 's/\.yaml$//' | tr '\n' ' ')"
  printf '%s' "$path"
}

run_profile() {
  local profile="$1"; shift || true
  local config
  config="$(profile_config "$profile")"

  check_docker
  require_gpu
  ensure_image

  [[ -d "$DATA_DIR/raw/findsum/text" ]] || die "dataset ausente. Rode: ./deploy.sh data"
  [[ -f "$MANIFEST" ]] || die "particao ausente. Rode: ./deploy.sh data"

  mkdir -p "$OUT_DIR/$profile"
  local logfile="$OUT_DIR/$profile/run.log"

  log "perfil: $profile ($config)"
  log "log: $logfile"
  log "a primeira execucao baixa os pesos do modelo (~6 GB) para $HF_CACHE"
  echo

  local started; started="$(date +%s)"
  # tee para que o log fique no host mesmo se a sessao SSH cair.
  if ! run_in_container findsum run --config "$config" "$@" 2>&1 | tee "$logfile"; then
    die "a execucao falhou. Veja $logfile"
  fi
  local elapsed=$(( $(date +%s) - started ))

  log "concluido em $((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m"
  compare_profile "$profile" || warn "a comparacao estatistica falhou (rode ./deploy.sh compare $profile)"
  collect "$profile"
  pack "$profile"
  report "$profile"
}

compare_profile() {
  local profile="$1"
  local dir="$OUT_DIR/$profile"
  [[ -d "$dir" ]] || die "nao ha resultados em $dir"
  ensure_image
  log "comparacao estatistica (Wilcoxon emparelhado + Holm-Bonferroni)"
  run_in_container findsum compare --run-dir "outputs/$profile" --reference C5 \
    | tee "$dir/comparison.txt"
}

# Copia para results/ apenas o que e leve o bastante para versionar.
collect() {
  local profile="$1"
  local src="$OUT_DIR/$profile"
  local dst="$RESULTS_DIR/$profile"
  [[ -d "$src" ]] || die "nao ha resultados em $src"

  rm -rf "$dst"; mkdir -p "$dst"
  local f
  for f in "${LIGHT_ARTIFACTS[@]}"; do
    [[ -f "$src/$f" ]] && cp "$src/$f" "$dst/"
  done
  local arm
  for arm in "$src"/*/; do
    [[ -f "$arm/scores.csv" ]] || continue
    mkdir -p "$dst/$(basename "$arm")"
    cp "$arm/scores.csv" "$dst/$(basename "$arm")/"
    [[ -f "$arm/summary.json" ]] && cp "$arm/summary.json" "$dst/$(basename "$arm")/"
  done
  log "results/$profile: $(du -sh "$dst" | cut -f1) (versionavel no git)"
}

pack() {
  local profile="$1"
  local src="$OUT_DIR/$profile"
  [[ -d "$src" ]] || die "nao ha resultados em $src"
  mkdir -p "$DIST_DIR"
  local tarball="$DIST_DIR/${profile}.tar.gz"
  tar -czf "$tarball" -C "$OUT_DIR" "$profile"
  log "dist/${profile}.tar.gz: $(du -sh "$tarball" | cut -f1)"
}

report() {
  local profile="$1"
  cat <<EOF

────────────────────────────────────────────────────────────────────
ONDE ESTA A SAIDA

  PARA PUBLICAR NO GIT (leve, ~50 KB):
    results/$profile/
      summary.json              metricas agregadas de cada configuracao
      reference_baseline.json   calibracao das metricas de ancoragem
      comparison.txt            tabela estatistica
      config.yaml               configuracao efetiva
      C*/scores.csv             metricas por documento

    git add results/$profile && git commit -m "resultados: $profile" && git push

  PARA BAIXAR E ANALISAR AQUI (completo, inclui os resumos gerados):
    dist/$profile.tar.gz

    scp <servidor>:$REPO_ROOT/dist/$profile.tar.gz .
    tar -xzf $profile.tar.gz

  COMPLETO NO SERVIDOR:
    outputs/$profile/
      C*/predictions.jsonl      cada resumo gerado, sua referencia e os
                                exemplos usados no prompt
      run.log                   log da execucao
────────────────────────────────────────────────────────────────────
EOF
}

# --------------------------------------------------------------------- despacho

usage() {
  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    ""|-h|--help|help) usage ;;
    doctor)  doctor ;;
    build)   build ;;
    data)    fetch_data ;;
    test)    ensure_image; run_in_container pytest -m "not slow" ;;
    shell)   ensure_image; run_in_container bash ;;
    run)     shift; [[ $# -ge 1 ]] || die "informe o perfil: ./deploy.sh run test_50"; run_profile "$@" ;;
    compare) shift; [[ $# -ge 1 ]] || die "informe o perfil"; compare_profile "$1"; collect "$1" ;;
    pack)    shift; [[ $# -ge 1 ]] || die "informe o perfil"; pack "$1" ;;
    collect) shift; [[ $# -ge 1 ]] || die "informe o perfil"; collect "$1" ;;
    *)
      # `./deploy.sh test_50` equivale a `./deploy.sh run test_50`.
      if [[ -f "$REPO_ROOT/configs/${cmd}.yaml" ]]; then
        run_profile "$@"
      else
        die "comando ou perfil desconhecido: '$cmd'. Use ./deploy.sh --help"
      fi
      ;;
  esac
}

main "$@"
