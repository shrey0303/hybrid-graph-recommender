"""
Unit tests for stream processor and model registry.

Tests cover:
- StreamEvent creation and serialization
- EventProcessor buffering and flushing
- Handler registration and callbacks
- KafkaStreamConfig generation
- ModelRegistry version management
- Active model switching
- Recommendation fallback
"""

import time

import pytest

from src.serving.stream_processor import (
    StreamEvent,
    EventProcessor,
    KafkaStreamConfig,
)
from src.serving.model_registry import ModelRegistry, ModelVersion


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_create_event(self):
        """Test event creation."""
        event = StreamEvent(
            event_id="e1",
            user_id="u1",
            item_id="i1",
            event_type="purchase",
            timestamp=time.time(),
        )
        assert event.user_id == "u1"
        assert event.event_type == "purchase"

    def test_event_serialization(self):
        """Test event to/from dict."""
        event = StreamEvent(
            event_id="e1", user_id="u1", item_id="i1",
            event_type="click", timestamp=1234567890.0,
            metadata={"source": "homepage"},
        )
        d = event.to_dict()
        assert d["event_id"] == "e1"
        assert d["metadata"]["source"] == "homepage"

        restored = StreamEvent.from_dict(d)
        assert restored.user_id == event.user_id
        assert restored.metadata == event.metadata


class TestEventProcessor:
    """Tests for EventProcessor."""

    def test_process_single_event(self):
        """Test processing a single event."""
        processor = EventProcessor(buffer_size=10)
        event = StreamEvent("e1", "u1", "i1", "click", time.time())

        result = processor.process_event(event)
        assert result["status"] == "processed"
        assert result["buffer_size"] == 1
        assert processor.processed_count == 1

    def test_buffer_fills_and_flushes(self):
        """Test that buffer auto-flushes at capacity."""
        processor = EventProcessor(buffer_size=3)
        flushed = False

        for i in range(3):
            event = StreamEvent(f"e{i}", "u1", f"i{i}", "click", time.time())
            result = processor.process_event(event)
            if result["flushed"]:
                flushed = True

        assert flushed, "Buffer should have flushed at capacity"
        assert "u1" not in processor.user_buffers

    def test_multiple_users(self):
        """Test separate buffers for different users."""
        processor = EventProcessor(buffer_size=10)

        processor.process_event(StreamEvent("e1", "u1", "i1", "click", time.time()))
        processor.process_event(StreamEvent("e2", "u2", "i2", "click", time.time()))

        assert len(processor.user_buffers) == 2
        assert len(processor.user_buffers["u1"]) == 1
        assert len(processor.user_buffers["u2"]) == 1

    def test_register_handler(self):
        """Test event handler registration and invocation."""
        processor = EventProcessor(buffer_size=10)
        handled_events = []

        processor.register_handler(
            "purchase", lambda e: handled_events.append(e)
        )

        event = StreamEvent("e1", "u1", "i1", "purchase", time.time())
        processor.process_event(event)

        assert len(handled_events) == 1
        assert handled_events[0].event_type == "purchase"

    def test_flush_callback(self):
        """Test on_flush callback."""
        processor = EventProcessor(buffer_size=2)
        flushed_data = []

        processor.on_flush(
            lambda uid, events: flushed_data.append((uid, len(events)))
        )

        processor.process_event(StreamEvent("e1", "u1", "i1", "click", time.time()))
        processor.process_event(StreamEvent("e2", "u1", "i2", "click", time.time()))

        assert len(flushed_data) == 1
        assert flushed_data[0] == ("u1", 2)

    def test_flush_all(self):
        """Test flushing all user buffers."""
        processor = EventProcessor(buffer_size=100)

        for uid in ["u1", "u2", "u3"]:
            processor.process_event(
                StreamEvent("e", uid, "i1", "click", time.time())
            )

        count = processor.flush_all()
        assert count == 3
        assert len(processor.user_buffers) == 0

    def test_process_batch(self):
        """Test batch event processing."""
        processor = EventProcessor(buffer_size=100)
        events = [
            StreamEvent(f"e{i}", f"u{i%3}", f"i{i}", "click", time.time())
            for i in range(9)
        ]

        result = processor.process_batch(events)
        assert result["total_processed"] == 9
        assert result["active_users"] == 3

    def test_event_weights(self):
        """Test that event weights are applied correctly."""
        processor = EventProcessor()
        event = StreamEvent("e1", "u1", "i1", "purchase", time.time())
        result = processor.process_event(event)
        assert result["weight"] == 1.0  # Purchase = highest weight

    def test_statistics(self):
        """Test processor statistics."""
        processor = EventProcessor()
        processor.process_event(StreamEvent("e1", "u1", "i1", "click", time.time()))

        stats = processor.get_statistics()
        assert stats["total_processed"] == 1
        assert stats["active_users"] == 1

    def test_repr(self):
        """Test string representation."""
        processor = EventProcessor()
        assert "EventProcessor" in repr(processor)


class TestKafkaStreamConfig:
    """Tests for Kafka configuration."""

    def test_default_config(self):
        """Test default Kafka config values."""
        config = KafkaStreamConfig()
        assert config.bootstrap_servers == "localhost:9092"
        assert config.input_topic == "user-interactions"

    def test_consumer_config(self):
        """Test consumer config generation."""
        config = KafkaStreamConfig(consumer_group="test-group")
        consumer = config.to_consumer_config()
        assert consumer["group_id"] == "test-group"
        assert "value_deserializer" in consumer

    def test_producer_config(self):
        """Test producer config generation."""
        config = KafkaStreamConfig()
        producer = config.to_producer_config()
        assert "value_serializer" in producer
        assert producer["acks"] == "all"

    def test_repr(self):
        assert "KafkaStreamConfig" in repr(KafkaStreamConfig())


class TestModelRegistry:
    """Tests for model registry."""

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry()
        version = registry.register(
            "v1.0", model=None, model_type="hybrid",
            metrics={"ndcg@10": 0.75},
        )
        assert version.version_id == "v1.0"
        assert len(registry.versions) == 1

    def test_set_active(self):
        """Test setting active model version."""
        registry = ModelRegistry()
        registry.register("v1.0", model=None)
        registry.register("v2.0", model=None)

        registry.set_active("v1.0")
        assert registry.active_version_id == "v1.0"
        assert registry.versions["v1.0"].is_active

        registry.set_active("v2.0")
        assert registry.active_version_id == "v2.0"
        assert not registry.versions["v1.0"].is_active
        assert registry.versions["v2.0"].is_active

    def test_set_active_nonexistent_raises(self):
        """Test that activating nonexistent version raises."""
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.set_active("v999")

    def test_fallback_recommendations(self):
        """Test fallback recs when no model loaded."""
        registry = ModelRegistry()
        recs = registry.recommend("user_1", num_items=5)
        assert len(recs) == 5
        assert all("title" in r for r in recs)  
        assert all("score" in r for r in recs)

    def test_list_versions(self):
        """Test listing all versions."""
        registry = ModelRegistry()
        registry.register("v1.0", model=None, metrics={"acc": 0.9})
        registry.register("v2.0", model=None, metrics={"acc": 0.95})

        versions = registry.list_versions()
        assert len(versions) == 2

    def test_serving_stats(self):
        """Test serving statistics."""
        registry = ModelRegistry()
        registry.recommend("u1")
        registry.recommend("u2")

        stats = registry.get_serving_stats()
        assert stats["total_inferences"] == 2
        assert "avg_latency_ms" in stats

    def test_repr(self):
        registry = ModelRegistry()
        assert "ModelRegistry" in repr(registry)
