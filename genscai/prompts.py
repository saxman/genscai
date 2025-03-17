from sqlalchemy import (
    String,
    Integer,
    PickleType,
    Column,
    create_engine,
    ForeignKey,
    desc,
)
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("prompts.id"))
    prompt = Column(String, nullable=False)
    model_id = Column(String)
    version = Column(Integer)
    mutation_prompt = Column(String)
    reasoning_prompt = Column(String)
    metrics = Column(PickleType)


class PromptCatalog:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def __del__(self):
        self.session.close()

    def store_prompt(self, prompt: Prompt):
        try:
            self.session.add(prompt)
        except:
            self.session.rollback()
            raise
        else:
            self.session.commit()

    def retrieve_last(self, model_id: str) -> Prompt:
        return self.session.query(Prompt).filter(Prompt.model_id == model_id).order_by(desc(Prompt.version)).first()

    def retrieve_all(self, model_id: str) -> list[Prompt]:
        return self.session.query(Prompt).filter(Prompt.model_id == model_id).order_by(desc(Prompt.version)).all()

    def delete_all(self, model_id: str) -> int:
        rows_deleted = 0

        try:
            rows_deleted = self.session.query(Prompt).filter(Prompt.model_id == model_id).delete()
        except:
            self.session.rollback()
            raise
        else:
            self.session.commit()

        return rows_deleted

    def retrieve_model_ids(self):
        return [x.model_id for x in self.session.query(Prompt.model_id).distinct()]
