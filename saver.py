from datetime import datetime
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Protocol, runtime_checkable

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    plate = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"Event(id={self.id}, plate={self.plate})"


@runtime_checkable
class Saver(Protocol):
    def save(self, data: Event) -> None: ...


class SQLSaver:
    # 'sqlite:///users.db'
    def __init__(self, sql_path: str):
        self.engine = create_engine(sql_path)
        Base.metadata.create_all(self.engine)

    def save(self, data: Event):
        with sessionmaker(bind=self.engine)() as session:
            try:
                session.add(data)
                session.commit()
            except Exception as e:
                session.rollback()
                print(e)
                # add logger later
            finally:
                session.close()


class SaverEngine:
    def __init__(self, saver: Saver, data_queue: Queue) -> None:
        if not isinstance(saver, Saver):
            raise ValueError(
                f"saver must be instance of {type(Saver)} found {type(saver)}"
            )

        self.saver = saver
        self.queue = data_queue
        self._stop = False

        self.thread = Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        while not self._stop:
            try:
                plate = self.queue.get(timeout=1)
            except Empty:
                continue

            if not isinstance(plate, str):
                raise ValueError(f"data must be str not {type(plate)}")
            new_event = Event(plate=plate)
            print("save", plate)
            self.saver.save(new_event)

    def stop(self):
        self._stop = True
        self.thread.join()


if __name__ == "__main__":
    saver = SQLSaver("sqlite:///events.db")
    event = Event(plate="K852CT18")

    saver.save(event)
