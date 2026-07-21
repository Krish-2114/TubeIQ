from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True)
    channel_id = Column(String, unique=True)
    title = Column(String)
    description = Column(Text, nullable=True)
    custom_url = Column(String, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    subscriber_count = Column(BigInteger, nullable=True)
    view_count = Column(BigInteger, nullable=True)
    video_count = Column(Integer, nullable=True)
    country = Column(String, nullable=True)
    niche = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True)
    last_fetched = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    videos = relationship("Video", back_populates="channel")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, ForeignKey("channels.id"))
    video_id = Column(String, unique=True)
    title = Column(String)
    description = Column(Text, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    view_count = Column(BigInteger, nullable=True)
    like_count = Column(BigInteger, nullable=True)
    comment_count = Column(BigInteger, nullable=True)
    tags = Column(Text, nullable=True)
    category_id = Column(String, nullable=True)
    engagement_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    channel = relationship("Channel", back_populates="videos")
