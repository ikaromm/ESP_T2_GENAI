from findsum_rag.examples import Example
from findsum_rag.grounding_audit import audit_amounts


def test_example_number_transfer_and_invented_scale_are_flagged():
    warnings = audit_amounts(
        'purchases of $268.3 million and technology of $199 million',
        'technology | 199 & 199,000 (2019)',
        'technology | 199 & 199,000 (2019)',
        [Example('other', 'other company', 'purchases of $268.3 million')],
    )
    assert [w['kind'] for w in warnings] == [
        'example_only_number', 'unsupported_scaled_amount',
    ]
    assert warnings[0]['example_ids'] == ['other']
    assert all(w['requires_human_review'] for w in warnings)


def test_number_shared_by_target_and_example_is_not_called_leakage():
    assert audit_amounts('$2 million', '$2,000,000', '$2,000,000',
                         [Example('other', '', '$2 million')]) == []
