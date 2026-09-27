"""Similaridade textual, coincidencia numerica e extratividade.

Coincidencia de valores nao avalia entidade, periodo ou relacao contabil.
Mesmo numeros trocados entre receita e lucro podem obter escore 1.
Avaliacao factual exige revisao de afirmacoes e evidencias (modulo factual).
Patamares da referencia sao diagnosticos da fonte, nunca tetos de desempenho.
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
    """Fracao de n-gramas copiados da fonte; extratividade, nao qualidade."""
    pred = _ngrams(_tokens(prediction), n)
    total = sum(pred.values())
    if not total:
        return 0.0
    src = _ngrams(_tokens(source), n)
    return _overlap(pred, src) / total


def reference_baseline(
    references: list[str], sources: list[str], *, tolerance: float = 1e-3
) -> dict[str, float]:
    """Diagnostico das referencias contra a fonte disponivel, nao teto.

    O maximo matematico de numeric_grounding continua sendo 1. Uma referencia
    pode incluir fatos ausentes da fonte distribuida. Extratividade nao tem
    direcao normativa: proximidade da referencia nao prova melhor qualidade.
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

        self._scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def score(self, prediction: str, reference: str) -> RougeScores:
        result = self._scorer.score(reference, prediction)
        return RougeScores(
            rouge1=result["rouge1"].fmeasure,
            rouge2=result["rouge2"].fmeasure,
            rougeL=result["rougeL"].fmeasure,
        )


BERTSCORE_MODEL = "xlnet-base-cased"


def _audit_bert_tokenizer(texts, tokenizer):
    from bert_score.utils import sent_encode
    from transformers import GPT2Tokenizer, RobertaTokenizer

    extra = (
        {"add_prefix_space": True}
        if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))
        else {}
    )
    encoded = {
        text: tokenizer.encode(text.strip(), add_special_tokens=True, truncation=False, **extra)
        for text in dict.fromkeys(texts)
    }
    # XLNet usa posicoes relativas e um sentinela ~1e30 no tokenizer. Em
    # transformers 5, esse sentinela excede o inteiro do backend. O limite
    # operacional e o MAIOR texto real, nao um corte predefinido da avaliacao.
    if tokenizer.model_max_length > 1_000_000_000:
        tokenizer.model_max_length = max((len(ids) for ids in encoded.values()), default=8)
    cache = {}
    for text, full in encoded.items():
        used = sent_encode(tokenizer, text)
        if full != used:
            raise ValueError(f"BERTScore truncaria/alteraria entrada: {len(full)} -> {len(used)}")
        cache[text] = {"full_tokens": len(full), "evaluated_tokens": len(used), "truncated": False}
    return [cache[text] for text in texts]


def bertscore_token_audit(texts: list[str], model_type: str = BERTSCORE_MODEL) -> list[dict]:
    """Confere os IDs integrais contra o tokenizer realmente usado pelo BERTScore."""
    from bert_score.utils import get_tokenizer

    return _audit_bert_tokenizer(texts, get_tokenizer(model_type, use_fast=False))


def bertscore_components(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = BERTSCORE_MODEL,
    batch_size: int = 1,
    device: str | None = None,
) -> list[PRF]:
    """Precisao, recall e F1 integrais; rejeita qualquer corte de tokens."""
    if len(predictions) != len(references):
        raise ValueError("predicoes e referencias devem ter o mesmo tamanho")
    if not predictions:
        return []
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
    )
    _audit_bert_tokenizer(predictions + references, scorer._tokenizer)
    precision, recall, f1 = scorer.score(
        predictions, references, batch_size=batch_size, verbose=False
    )
    return [
        PRF(float(p), float(r), float(f)) for p, r, f in zip(precision, recall, f1, strict=True)
    ]


def bertscore(predictions, references, **kwargs) -> list[float]:
    """API historica de F1; mesma avaliacao integral dos componentes."""
    return [s.f1 for s in bertscore_components(predictions, references, **kwargs)]


def ensure_meteor_resources():
    """Falha explicita sem WordNet; nao baixa recursos durante a avaliacao."""
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()


def meteor(prediction: str, reference: str) -> float:
    """METEOR NLTK, Treebank, lowercase, Porter/WordNet e parametros padrao."""
    from nltk.tokenize import TreebankWordTokenizer
    from nltk.translate.meteor_score import single_meteor_score

    ensure_meteor_resources()
    tokenizer = TreebankWordTokenizer()
    return float(
        single_meteor_score(
            tokenizer.tokenize(reference.lower()),
            tokenizer.tokenize(prediction.lower()),
            alpha=0.9,
            beta=3.0,
            gamma=0.5,
        )
    )


SUPPLEMENTARY_METRICS = {
    "version": 1,
    "scope": "descriptive_only_not_added_to_primary_hypothesis_tests",
    "metrics": ["bertscore_precision", "bertscore_recall", "meteor"],
    "bertscore": {"model": BERTSCORE_MODEL, "layer": 5, "idf": False, "rescale": False},
    "meteor": {
        "implementation": "nltk.translate.meteor_score.single_meteor_score",
        "tokenizer": "TreebankWordTokenizer",
        "lowercase": True,
        "stemmer": "PorterStemmer",
        "synonyms": "WordNet English",
        "alpha": 0.9,
        "beta": 3.0,
        "gamma": 0.5,
    },
}


@dataclass
class DocumentScores:
    """Todas as metricas de um unico resumo gerado."""

    doc_id: str
    rouge: RougeScores
    n_words: int
    numeric: PRF | None = None
    numeric_grounding: float | None = None
    ngram_grounding: float | None = None
    bertscore: float | None = None
    extra: dict = field(default_factory=dict)

    def flat(self) -> dict[str, float | str | None]:
        """Formato tabular, para agregacao e gravacao em CSV."""
        row = {
            "doc_id": self.doc_id,
            "rouge1": self.rouge.rouge1,
            "rouge2": self.rouge.rouge2,
            "rougeL": self.rouge.rougeL,
            "bertscore": self.bertscore,
            "n_words": self.n_words,
            **self.extra,
        }
        if self.numeric is not None:
            row.update(
                numeric_precision=self.numeric.precision,
                numeric_recall=self.numeric.recall,
                numeric_f1=self.numeric.f1,
                numeric_grounding=self.numeric_grounding,
                ngram_grounding=self.ngram_grounding,
            )
        return row


def score_document(
    doc_id: str,
    prediction: str,
    reference: str,
    source: str,
    *,
    rouge: RougeScorer | None = None,
    legacy_diagnostics: bool = False,
) -> DocumentScores:
    """Calcula ROUGE; diagnosticos numericos legados exigem opt-in explicito.

    BERTScore fica de fora porque compensa rodar em lote; use `bertscore()` e
    preencha o campo depois.
    """
    scorer = rouge or RougeScorer()
    return DocumentScores(
        doc_id=doc_id,
        rouge=scorer.score(prediction, reference),
        numeric=numeric_prf(prediction, reference) if legacy_diagnostics else None,
        numeric_grounding=numeric_grounding(prediction, source) if legacy_diagnostics else None,
        ngram_grounding=ngram_grounding(prediction, source) if legacy_diagnostics else None,
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
