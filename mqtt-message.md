Topic: "skyscan/config/json"

The JSON blob has different config values. There are no required fields. The following Keys are used:
- cameraZoom - int value from 0 - 9999
- cameraDelay - float value for the number of seconds to wait after sending the camera move API command, before sending the take picture API command.
- cameraMoveSpeed - This is how fast the camea will move. It is an Int number between 0-99 (max)
- cameraLead - This is how far the tracker should move the center point in front of the currently tracked plane. It is a float, and is measured in seconds, example: 0.25 . It is based on the planes current heading and how fast it is going. 