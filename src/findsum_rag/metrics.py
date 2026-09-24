"""Metricas de avaliacao nas tres dimensoes do estudo.

* **Qualidade textual** -- ROUGE-1/2/L e BERTScore contra o resumo de referencia.
* **Preservacao numerica** -- precisao, cobertura e F1 dos numeros do resumo
  gerado em relacao ao resumo de referencia.
* **Fidelidade factual** -- ancoragem dos numeros do resumo gerado no documento
  de origem. Um numero que nao esta no documento e alucinacao, independentemente
  de estar ou nao na referencia.
* **Extratividade** -- sobreposicao de n-gramas com o documento. Nao e fidelidade:
  serve para detectar colagem e deve ser lida contra o valor da propria
  referencia (ver `reference_baseline`).

A distincao entre as duas primeiras comparacoes e deliberada: comparar com a
referencia mede se o resumo cobre o que deveria; comparar com o documento mede se
o resumo inventa. Um resumo pode ir bem em uma e mal na outra.

Todas as metricas de ancoragem precisam de calibracao. Nos dados reais o proprio
resumo de referencia marca `numeric_grounding` ~0.56 e `ngram_grounding` ~0.12
contra o texto distribuido, e sao esses os patamares de comparacao -- nao 1.0.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Captura numeros com separador de milhar e decimais, com sinal opcional.
# O FINDSum e distribuido tokenizado ("$ 50.0 million", "12.5 %"), por isso o
# simbolo de moeda nao faz parte do numero.
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")

# Escalas textuais que multiplicam o valor escrito.
SCALES = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
}


@dataclass(frozen=True)
class NumberMention:
    """Um numero encontrado no texto, com a forma original e o valor normalizado."""

    raw: str
    value: float
    scaled: bool = False


def extract_numbers(text: str, *, apply_scale: bool = True) -> list[NumberMention]:
    """Extrai os numeros de um texto.

    Quando `apply_scale` e verdadeiro, uma escala textual imediatamente posterior
    ao numero (`million`, `billion`, ...) e incorporada ao valor, de modo que
    "$ 50.0 million" e "50,000,000" sejam reconhecidos como o mesmo valor.

    Percentuais nao recebem tratamento especial: "12.5 %" produz o valor 12.5, o
    que e o desejado para comparar percentuais entre si.
    """
    mentions: list[NumberMention] = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - a regex garante o formato
            continue
        scaled = False
        if apply_scale:
            tail = text[match.end() : match.end() + 24].lower()
            word = re.match(r"\s*([a-z]+)", tail)
            if word and word.group(1) in SCALES:
                value *= SCALES[word.group(1)]
                scaled = True
        mentions.append(NumberMention(raw=raw, value=value, scaled=scaled))
    return mentions


def _value_counter(text: str, *, tolerance: float) -> Counter:
    """Conta os numeros de um texto por valor arredondado.

    O arredondamento relativo absorve diferencas de formatacao ("1.7" vs
    "1,653,732" quando acompanhados de escala) sem colapsar valores distintos.
    """
    counter: Counter = Counter()
    for mention in extract_numbers(text):
        counter[_bucket(mention.value, tolerance)] += 1
    return counter


def _bucket(value: float, tolerance: float) -> float:
    """Discretiza um valor para comparacao com tolerancia relativa."""
    if value == 0 or tolerance <= 0:
        return round(value, 4)
    import math

    magnitude = math.floor(math.log10(abs(value)))
    step = max(abs(value) * tolerance, 10.0 ** (magnitude - 4))
    return round(value / step) * step


@dataclass
class PRF:
    """Tripla precisao / cobertura / F1."""

    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, matched: int, predicted: int, expected: int) -> PRF:
        precision = matched / predicted if predicted else 0.0
        recall = matched / expected if expected else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return cls(precision=precision, recall=recall, f1=f1)


def _overlap(a: Counter, b: Counter) -> int:
    """Tamanho da interseccao de dois multiconjuntos."""
    return sum(min(count, b[key]) for key, count in a.items())


def numeric_prf(prediction: str, reference: str, *, tolerance: float = 1e-3) -> PRF:
    """Preservacao numerica do resumo gerado em relacao a referencia."""
    pred = _value_counter(prediction, tolerance=tolerance)
    ref = _value_counter(reference, tolerance=tolerance)
    return PRF.from_counts(_overlap(pred, ref), sum(pred.values()), sum(ref.values()))


def numeric_grounding(prediction: str, source: str, *, tolerance: float = 1e-3) -> float:
    """Fracao dos numeros do resumo que existem no documento de origem.

    Devolve 1.0 quando o resumo nao contem numeros -- sem numeros nao ha numero
    alucinado. Interprete sempre junto de `numeric.recall`, que penaliza resumos
    que simplesmente evitam citar valores.
    """
    pred = _value_counter(prediction, tolerance=tolerance)
    total = sum(pred.values())
    if not total:
        return 1.0
    src = _value_counter(source, tolerance=tolerance)
    return _overlap(pred, src) / total


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.,%$-]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def ngram_grounding(prediction: str, source: str, n: int = 4) -> float:
    """Fracao dos n-gramas do resumo que aparecem no documento de origem.

    **Isto mede extratividade, nao fidelidade.** Medido nos dados reais, o
    proprio resumo de referencia do FINDSum marca ~0.12 (tarefa Liquidity, split
    val), porque os resumos sao abstrativos: foram escritos, nao copiados. Um
    modelo que marcasse 0.9 aqui estaria colando trechos, o que e pior e nao
    melhor.

    Portanto interprete sempre contra o valor da referencia, dado por
    `reference_baseline()`: o que interessa e a PROXIMIDADE ao nivel de
    extratividade da referencia, nao o valor absoluto.
    """
    pred = _ngrams(_tokens(prediction), n)
    total = sum(pred.values())
    if not total:
        return 0.0
    src = _ngrams(_tokens(source), n)
    return _overlap(pred, src) / total


def reference_baseline(
    references: list[str], sources: list[str], *, tolerance: float = 1e-3
) -> dict[str, float]:
    """Calibra as metricas de ancoragem usando os proprios resumos de referencia.

    Sem esta calibracao as metricas de ancoragem sao ininterpretaveis. Nos dados
    reais a referencia marca `numeric_grounding` ~0.56, ou seja **~44% dos
    numeros do resumo humano nao estao no texto distribuido** -- consequencia da
    selecao de conteudo aplicada pelo FINDSum (ver `docs/dataset.md`). Isso
    estabelece um teto pratico: nenhum modelo que use so este texto como fonte
    deveria ser cobrado de ultrapassar esse patamar.

    Raises:
        ValueError: se as listas tiverem tamanhos diferentes.
    """
    if len(references) != len(sources):
        raise ValueError(f"{len(references)} referencias para {len(sources)} documentos")
    if not references:
        return {}

    import statistics

    numeric = [
        numeric_grounding(r, s, tolerance=tolerance)
        for r, s in zip(references, sources, strict=True)
    ]
    ngram = [ngram_grounding(r, s) for r, s in zip(references, sources, strict=True)]
    words = [len(r.split()) for r in references]
    return {
        "n_docs": float(len(references)),
        "reference_numeric_grounding_mean": statistics.fmean(numeric),
        "reference_numeric_grounding_median": statistics.median(numeric),
        "reference_ngram_grounding_mean": statistics.fmean(ngram),
        "reference_ngram_grounding_median": statistics.median(ngram),
        "reference_n_words_mean": statistics.fmean(words),
    }


@dataclass
class RougeScores:
    """ROUGE-1, ROUGE-2 e ROUGE-L (F-measure)."""

    rouge1: float
    rouge2: float
    rougeL: float


class RougeScorer:
    """Wrapper fino sobre `rouge_score`, com stemming ligado."""

    def __init__(self) -> None:
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(self, prediction: str, reference: str) -> RougeScores:
        result = self._scorer.score(reference, prediction)
        return RougeScores(
            rouge1=result["rouge1"].fmeasure,
            rouge2=result["rouge2"].fmeasure,
            rougeL=result["rougeL"].fmeasure,
        )


def bertscore(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = "roberta-large",
    batch_size: int = 16,
    device: str | None = None,
) -> list[float]:
    """BERTScore F1 por par. Calculado em lote por ser custoso em GPU."""
    if not predictions:
        return []
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(v) for v in f1]


@dataclass
class DocumentScores:
    """Todas as metricas de um unico resumo gerado."""

    doc_id: str
    rouge: RougeScores
    numeric: PRF
    numeric_grounding: float
    ngram_grounding: float
    n_words: int
    bertscore: float | None = None
    extra: dict = field(default_factory=dict)

    def flat(self) -> dict[str, float | str | None]:
        """Formato tabular, para agregacao e gravacao em CSV."""
        return {
            "doc_id": self.doc_id,
            "rouge1": self.rouge.rouge1,
            "rouge2": self.rouge.rouge2,
            "rougeL": self.rouge.rougeL,
            "bertscore": self.bertscore,
            "numeric_precision": self.numeric.precision,
            "numeric_recall": self.numeric.recall,
            "numeric_f1": self.numeric.f1,
            "numeric_grounding": self.numeric_grounding,
            "ngram_grounding": self.ngram_grounding,
            "n_words": self.n_words,
        }


def score_document(
    doc_id: str,
    prediction: str,
    reference: str,
    source: str,
    *,
    rouge: RougeScorer | None = None,
) -> DocumentScores:
    """Calcula todas as metricas deterministicas de um resumo.

    BERTScore fica de fora porque compensa rodar em lote; use `bertscore()` e
    preencha o campo depois.
    """
    scorer = rouge or RougeScorer()
    return DocumentScores(
        doc_id=doc_id,
        rouge=scorer.score(prediction, reference),
        numeric=numeric_prf(prediction, reference),
        numeric_grounding=numeric_grounding(prediction, source),
        ngram_grounding=ngram_grounding(prediction, source),
        n_words=len(prediction.split()),
    )


def aggregate(scores: list[DocumentScores]) -> dict[str, float]:
    """Media e mediana de cada metrica sobre um conjunto de documentos."""
    import statistics

    if not scores:
        return {}
    rows = [s.flat() for s in scores]
    summary: dict[str, float] = {"n_docs": float(len(rows))}
    for key in rows[0]:
        if key == "doc_id":
            continue
        values = [r[key] for r in rows if isinstance(r[key], (int, float))]
        if not values:
            continue
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_median"] = statistics.median(values)
    return summary
