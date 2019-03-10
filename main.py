from GenerateModule import GenerateAndTrainNN

from Params import GlobalSwMode
from SwModule import launchSw
def main():
    launchSw()
    """
    if GlobalSwMode == 'Train':
        GenerateAndTrainNN()

    if GlobalSwMode == 'LaunchSpawnedNN':
        print("Analyzer not yet supported")
    """

if __name__ == '__main__':
    main()

