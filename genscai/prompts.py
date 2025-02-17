import genscai

from sqlalchemy import String, Integer, Float, Column, create_engine, ForeignKey, desc
from sqlalchemy.orm import sessionmaker, Session, declarative_base

Base = declarative_base()


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("prompts.id"))
    prompt = Column(String, nullable=False)
    model_id = Column(String)
    version = Column(String)
    mutation_prompt = Column(String)
    metrics = Column(String)


class PromptCatalog:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def __del__(self):
        self.session.close()

    def store_prompt(self, prompt: Prompt):
        self.session.add(prompt)
        self.session.commit()

    def retrieve_latest(self, model_id: str) -> Prompt:
        return (
            self.session.query(Prompt)
            .filter(Prompt.model_id == model_id)
            .order_by(desc(Prompt.version))
            .limit(1)
            .one()
        )

    def retrieve_all(self, model_id: str) -> list[Prompt]:
        return (
            self.session.query(Prompt)
            .filter(Prompt.model_id == model_id)
            .order_by(desc(Prompt.version))
            .all()
        )

    def delete_all(self, model_id: str):
        try:
            num_rows_deleted = (
                self.session.query(Prompt).filter(Prompt.model_id == model_id).delete()
            )
            print(f"deleting {num_rows_deleted} rows")
            self.session.commit()
        except:
            self.session.rollback()
