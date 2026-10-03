from btg_barc import BARCReplayBuffer


buffer = BARCReplayBuffer(capacity=1000, seed=42)

for step in range(100):
    record_id = buffer.add(
        payload={"step": step},
        priority=(step + 1) / 100,
        novelty=max(0.0, 1.0 - step / 100),
        risk=0.05,
    )

batch = buffer.sample(8)

for sample in batch:
    print(
        sample.record_id,
        round(sample.probability, 6),
        round(sample.importance_weight, 6),
        round(sample.priority_share, 4),
    )
    # After a learner update:
    buffer.update_priority(sample.record_id, priority=0.25)

print(buffer.controller_state())
