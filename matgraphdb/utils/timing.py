import functools
import time

class Timer:
    """A class to keep track of events and time in a code"""
    def __init__(self):
        """
        start_time : start time of Timer
        events : List[Tuple(3)] of (event_name: str, event_elapsed_time, total_event_time)
        """
        self.start_time=time.time()

        self.events=[]
        self.events_detials = ("Event Name", "Event Elapsed Time", "Total Event Time")
        self.current_event=None

    def start_event(self, event_name):
        """
        A method to record an event. This method will keep tack of the ekapsed 
        time and name of the event record will 
        """ 
        if self.current_event:
            raise ("End the event before starting a new one")
        self.current_event = event_name
        self.event_elapsed_time = time.time()

    def end_event(self):
        """
        A method to end a the recording of an event.
        """

        self.event_elapsed_time = time.time()-self.event_elapsed_time
        total_elapsed_time = time.time()-self.start_time 

        self.events.append((self.current_event,self.event_elapsed_time,total_elapsed_time))

        self.current_event = None
        self.event_elapsed_time = None

    def save_events(self,filename, delimiter='       ,       '):
        col1_width = 30
        col2_width = 25
        col3_width = 25
        with open(filename,'w') as file:
            file.write("Events log\n")
            file.write("-----------------------------------------------------------------------\n")
            file.write("-----------------------------------------------------------------------\n")

            event_details_str =  delimiter.join(self.events_detials)
            file.write("{:<30}{:<30}{:<30}\n".format(*self.events_detials))
            # file.write(f"{event_details_str}\n")
            file.write("-----------------------------------------------------------------------\n")

            for event in self.events:

                event_str = [str(item) for item in event]
                # event_str =  delimiter.join(event_str)

                # file.write(f"{event_str}\n")
                file.write("{:<30}{:<30}{:<30}\n".format(*event_str))
            


def timeit(func):
    """A decorator that times a function's execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__!r} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper