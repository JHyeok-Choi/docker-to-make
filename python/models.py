from sqlalchemy import Column, String, INT, TEXT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class St_info(Base):
    __tablename__ = 'st_info'

    ST_ID = Column(INT, nullable=False, primary_key=True)
    NAME = Column(TEXT, nullable=False)
    DEPT = Column(TEXT, nullable=False)

class St_grade(Base):
    __tablename__ = 'st_grad'

    ST_ID = Column(INT, nullable=False, primary_key=True)
    Linux = Column(INT, nullable=False)
    DB = Column(INT, nullable=False)
