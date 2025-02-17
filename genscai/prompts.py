import genscai

Base = declarative_base()

class Prompt(Base):
    __tablename__ = 'prompts'
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey('prompts.id'), nullable=True)
    prompt = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    version = Column(String, nullable=False)

engine = create_engine('sqlite:///' / genscai.paths.data / 'prompt_catalog.db')
Base.metadata.create_all(engine)
