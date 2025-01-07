import functools
import logging
import time

logger = logging.getLogger(__name__)

class Timer:
    """A class to keep track of events and time in a code"""
    def __init__(self):
        """
        Initializes a Timing object.

        This method initializes the Timing object by setting the start time to the current time,
        initializing the events list and events_details tuple, and setting the current_event to None.
        """
        self.start_time = time.time()
        self.events = []
        self.events_details = ("Event Name", "Event Elapsed Time", "Total Event Time")
        self.current_event = None
    def __init__(self):
        
        self.start_time=time.time()

        self.events=[]
        self.events_detials = ("Event Name", "Event Elapsed Time", "Total Event Time")
        self.current_event=None

    def start_event(self, event_name):
        """
        Starts a new event and records the start time.

        Args:
            event_name (str): The name of the event.

        Raises:
            Exception: If there is already an ongoing event.

        """
        if self.current_event:
            raise Exception("End the event before starting a new one")
        self.current_event = event_name
        self.event_elapsed_time = time.time()

    def end_event(self):
        """
        Ends the current event and records the elapsed time.

        This method calculates the elapsed time for the current event and the total elapsed time since the start of the timer.
        It appends the event details (event name, event elapsed time, total elapsed time) to the list of events.

        Args:
            None

        Returns:
            None
        """
        self.event_elapsed_time = time.time() - self.event_elapsed_time
        total_elapsed_time = time.time() - self.start_time

        self.events.append((self.current_event, self.event_elapsed_time, total_elapsed_time))

        self.current_event = None
        self.event_elapsed_time = None

    def save_events(self, filename, delimiter='       ,       '):
        """
        Save the events log to a file.

        Args:
            filename (str): The name of the file to save the events log to.
            delimiter (str, optional): The delimiter to use between columns in the log. Defaults to '       ,       '.
        """
        col1_width = 30
        col2_width = 25
        col3_width = 25
        with open(filename, 'w') as file:
            file.write("Events log\n")
            file.write("-----------------------------------------------------------------------\n")
            file.write("-----------------------------------------------------------------------\n")

            event_details_str = delimiter.join(self.events_detials)
            file.write("{:<30}{:<30}{:<30}\n".format(*self.events_detials))
            file.write("-----------------------------------------------------------------------\n")

            for event in self.events:
                event_str = [str(item) for item in event]
                file.write("{:<30}{:<30}{:<30}\n".format(*event_str))
            


def timeit(func):
    """
    A decorator that measures the execution time of a function.

    Args:
        func: The function to be timed.

    Returns:
        The wrapped function.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"Function {func.__name__!r} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper