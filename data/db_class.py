from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from data.db_init import Base


class Model(Base):
    __tablename__ = "Model"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    version = Column(String, index=True)
    created_at = Column(String, index=True)
    path = Column(String, index=True)

    predictions = relationship("Prediction", back_populates="model")


class RealTemperature(Base):
    __tablename__ = "RealTemperature"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    temperature_2m = Column(String, index=True)
    relative_humidity = Column(String, index=True)
    precipitation = Column(String, index=True)
    surface_pressure = Column(String, index=True)
    latitude = Column(String, index=True)
    longitude = Column(String, index=True)

    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "latitude",
            "longitude",
            name="unique_timestamp_latitude_longitude",
        ),
    )


class Prediction(Base):
    __tablename__ = "Prediction"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("Model.id"))
    timestamp = Column(String, index=True)
    relative_humidity = Column(String, index=True)
    precipitation = Column(String, index=True)
    surface_pressure = Column(String, index=True)
    latitude = Column(String, index=True)
    longitude = Column(String, index=True)
    real = Column(String, index=True)
    prediction = Column(String, index=True)

    model = relationship("Model", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "latitude",
            "longitude",
            "model_id",
            name="unique_timestamp_latitude_longitude_model_id",
        ),
    )
