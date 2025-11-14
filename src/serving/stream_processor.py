"""
Kafka Event Stream Processor for Real-Time Recommendations.

Handles the streaming pipeline:
1. Consumes user interaction events from Kafka topics
2. Updates user embeddings in real-time
3. Triggers re-ranking when significant events occur
4. Publishes recommendation updates to downstream topics

Architecture:
    [User Events] → [Kafka Consumer] → [Event Processor] → [Model Update]
                                                           ↓
                                              [Kafka Producer] → [Rec Updates]
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class StreamEvent:
    """
    Represents a user interaction event from the stream.

    Attributes:
        event_id: Unique event identifier.
        user_id: User who performed the action.
        item_id: Item involved in the interaction.
        event_type: Type of event (view, click, purchase, rating).
        timestamp: Unix timestamp of the event.
        metadata: Additional event data.
    """
    event_id: str
    user_id: str
    item_id: str
    event_type: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(
            event_id=data["event_id"],
            user_id=data["user_id"],
            item_id=data["item_id"],
            event_type=data["event_type"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


class EventProcessor:
    """
    Process streaming events and update recommendation state.

    Maintains a buffer of recent events per user and triggers
    embedding updates when thresholds are reached.

    Attributes:
        buffer_size: Max events to buffer per user before flush.
        event_weights: Importance weights for different event types.
        user_buffers: Per-user event buffers.
        handlers: Registered event type handlers.
        processed_count: Total events processed.

    Example:
        >>> processor = EventProcessor(buffer_size=10)
        >>> processor.register_handler("purchase", handle_purchase)
        >>> processor.process_event(event)
    """

    # Event type importance weights
    DEFAULT_WEIGHTS = {
        "view": 0.1,
        "click": 0.3,
        "add_to_cart": 0.5,
        "purchase": 1.0,
        "rating": 0.8,
        "skip": -0.2,
    }

    def __init__(
        self,
        buffer_size: int = 50,
        event_weights: Optional[Dict[str, float]] = None,
        flush_interval_seconds: float = 30.0,
    ) -> None:
        """
        Initialize event processor.

        Args:
            buffer_size: Max events per user before auto-flush.
            event_weights: Custom event importance weights.
            flush_interval_seconds: Time between auto-flushes.
        """
        self.buffer_size = buffer_size
        self.event_weights = event_weights or self.DEFAULT_WEIGHTS
        self.flush_interval = flush_interval_seconds

        self.user_buffers: Dict[str, List[StreamEvent]] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.processed_count = 0
        self.last_flush_time = time.time()

        self._on_flush_callbacks: List[Callable] = []

        logger.info(
            f"EventProcessor initialized | "
            f"buffer_size={buffer_size}, flush_interval={flush_interval_seconds}s"
        )

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[StreamEvent], None],
    ) -> None:
        """
        Register a handler function for a specific event type.

        Args:
            event_type: Event type to handle (e.g., 'purchase').
            handler: Callback function receiving the event.
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event_type='{event_type}'")

    def on_flush(self, callback: Callable[[str, List[StreamEvent]], None]) -> None:
        """Register a callback that fires when a user buffer is flushed."""
        self._on_flush_callbacks.append(callback)

    def process_event(self, event: StreamEvent) -> Dict[str, Any]:
        """
        Process a single streaming event.

        Steps:
        1. Validate and weight the event
        2. Add to user buffer
        3. Call registered handlers
        4. Flush if buffer is full

        Args:
            event: The streaming event to process.

        Returns:
            Processing result with status and metrics.
        """
        self.processed_count += 1

        # Get event weight
        weight = self.event_weights.get(event.event_type, 0.1)

        # Add to user buffer
        if event.user_id not in self.user_buffers:
            self.user_buffers[event.user_id] = []

        self.user_buffers[event.user_id].append(event)

        # Call handlers
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event.event_type}: {e}")

        # Check buffer flush
        result = {
            "status": "processed",
            "event_type": event.event_type,
            "weight": weight,
            "buffer_size": len(self.user_buffers[event.user_id]),
            "flushed": False,
        }

        if len(self.user_buffers[event.user_id]) >= self.buffer_size:
            self._flush_user_buffer(event.user_id)
            result["flushed"] = True

        return result

    def process_batch(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """
        Process a batch of events.

        Args:
            events: List of events to process.

        Returns:
            Batch processing summary.
        """
        results = []
        for event in events:
            result = self.process_event(event)
            results.append(result)

        flushed_count = sum(1 for r in results if r["flushed"])

        return {
            "total_processed": len(events),
            "buffers_flushed": flushed_count,
            "active_users": len(self.user_buffers),
        }

    def _flush_user_buffer(self, user_id: str) -> None:
        """Flush a user's event buffer and trigger callbacks."""
        if user_id not in self.user_buffers:
            return

        events = self.user_buffers.pop(user_id)

        for callback in self._on_flush_callbacks:
            try:
                callback(user_id, events)
            except Exception as e:
                logger.error(f"Flush callback error for user {user_id}: {e}")

        logger.debug(
            f"Flushed {len(events)} events for user {user_id}"
        )

    def flush_all(self) -> int:
        """
        Flush all user buffers.

        Returns:
            Number of users flushed.
        """
        user_ids = list(self.user_buffers.keys())
        for uid in user_ids:
            self._flush_user_buffer(uid)

        self.last_flush_time = time.time()
        logger.info(f"Flushed all buffers ({len(user_ids)} users)")
        return len(user_ids)

    def get_user_history(self, user_id: str) -> List[StreamEvent]:
        """Get buffered events for a user."""
        return self.user_buffers.get(user_id, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        buffer_sizes = [
            len(events) for events in self.user_buffers.values()
        ]
        return {
            "total_processed": self.processed_count,
            "active_users": len(self.user_buffers),
            "avg_buffer_size": float(
                sum(buffer_sizes) / max(len(buffer_sizes), 1)
            ),
            "max_buffer_size": max(buffer_sizes, default=0),
            "registered_handlers": {
                k: len(v) for k, v in self.handlers.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"EventProcessor("
            f"processed={self.processed_count}, "
            f"active_users={len(self.user_buffers)})"
        )


class KafkaStreamConfig:
    """
    Configuration for Kafka streaming integration.

    Defines topic names, consumer groups, and serialization settings
    for the recommendation event pipeline.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "recommendation-engine",
        input_topic: str = "user-interactions",
        output_topic: str = "recommendation-updates",
        feedback_topic: str = "user-feedback",
        auto_offset_reset: str = "latest",
        max_poll_records: int = 100,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.consumer_group = consumer_group
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.feedback_topic = feedback_topic
        self.auto_offset_reset = auto_offset_reset
        self.max_poll_records = max_poll_records

    def to_consumer_config(self) -> Dict[str, Any]:
        """Generate kafka-python consumer configuration."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.consumer_group,
            "auto_offset_reset": self.auto_offset_reset,
            "max_poll_records": self.max_poll_records,
            "value_deserializer": lambda m: json.loads(m.decode("utf-8")),
            "enable_auto_commit": True,
        }

    def to_producer_config(self) -> Dict[str, Any]:
        """Generate kafka-python producer configuration."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "acks": "all",
            "retries": 3,
        }

    def __repr__(self) -> str:
        return (
            f"KafkaStreamConfig("
            f"servers='{self.bootstrap_servers}', "
            f"input='{self.input_topic}', "
            f"output='{self.output_topic}')"
        )
