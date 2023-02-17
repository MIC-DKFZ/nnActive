import json

from pydantic.dataclasses import dataclass

from nnactive.results.utils import get_results_folder


@dataclass
class State:
    loop: int
    last_step: str 

    @classmethod
    def get_id_state(cls, id:int) ->State:
        fn = get_results_folder(id)/"state.json"
        with open(fn, "r") as file:
            file = json.load(fn)
        return State(**file)

    def save_state(self):
        fn = get_results_folder(id)
        with open(fn , "w") as file:
            # TODO: SAVING DATACLASS Delete pydantic key value pairs here
            json.dump(self.__dict__, file)

    
        

        