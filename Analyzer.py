import threading
import time

from DataSet import load_prediction_data as MaintainceLog

from LaunchTrainedNN import AnalyzerModule

class AnalyzerThread():
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, interval=1, callBack =None, Args =None, Predictor = None):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval
        self._predictor = Predictor
    """
        if callBack is None:
            thread = threading.Thread(target=self.run, args=())
        else:
            thread = threading.Thread(target=callBack, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    """
    def loadData(self):
        testData, expectLabel = MaintainceLog()

        print(testData)
        print("Expected labels")
        print(expectLabel)

        print("Predicting")
        predResult = self._predictor.run(testData)

        print(predResult)

        print("")
    def run(self):
        """ Method that runs forever """
        while True:
            # Do something
            print('Doing something imporant in the background')

            self.loadData()
            time.sleep(self.interval)
