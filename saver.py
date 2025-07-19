from datetime import datetime
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Protocol, runtime_checkable

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc

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


class SQLSession():
    def __init__(self, sql_path):
        self.engine = create_engine(sql_path)
        Base.metadata.create_all(self.engine)
        self.maker = sessionmaker(bind=self.engine)

    def __call__(self):
        return self.maker()

class SQLResults():
    def __init__(self, session, last=10):
        self.last = last
        self.session = session

    def __call__(self):
        results = (
            self.session.query(Event)
            .order_by(desc(Event.created_at))
            .limit(self.last)
            .all()
        )

        return [(result.plate, result.created_at)for result in results]

class SQLSaver:
    # 'sqlite:///users.db'
    def __init__(self, sql_session):
        
        self.SessionLocal = sql_session 

    def save(self, data: list[Event]):
        with self.SessionLocal as session:
            try:
                session.add_all(data)
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
            events = []
            while not self.queue.empty():
                plate = self.queue.get()
                if not isinstance(plate, str):
                    raise ValueError(f"data must be str not {type(plate)}")
                events.append(Event(plate=plate))

            self.saver.save(events)
            events.clear()
            sleep(0.1)  # пауза между итерациями

    def stop(self):
        self._stop = True
        self.thread.join()


if __name__ == "__main__":
    saver = SQLSaver("sqlite:///events.db")
    event = Event(plate="K852CT18")

    saver.save(event)
