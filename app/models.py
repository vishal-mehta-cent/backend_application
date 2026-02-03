from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=True)
    password = Column(String, nullable=True)  # For Google login, this can be null
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    orders = relationship("Order", back_populates="user")
    funds = relationship("Fund", back_populates="user", uselist=False)

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username"))
    script = Column(String)
    order_type = Column(String)  # BUY or SELL
    quantity = Column(Integer)
    price = Column(Float)
    status = Column(String, default="OPEN")  # OPEN, CLOSED
    stoploss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="orders")

class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username"))
    script = Column(String)
    quantity = Column(Integer)
    average_price = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Fund(Base):
    __tablename__ = "funds"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username"), unique=True)
    total_funds = Column(Float, default=0.0)
    available_funds = Column(Float, default=0.0)

    user = relationship("User", back_populates="funds")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    category = Column(String)  # feedback / contact
    name = Column(String)
    email = Column(String)
    phone = Column(String, nullable=True)
    message = Column(String)
    submitted_at = Column(DateTime, default=datetime.utcnow)
