"""This file include the parent class for all future edge technology work by IQT Labs that
incorporates a dynamic message/event-based infrastructure that is enabled via MQTT.
This is very much a working document and is under active development.
"""
import json
from typing import Callable, Any, Dict, List
import paho.mqtt.client as mqtt


class BaseMQTTPubSub:
    """The parent class for the core edge technology module under development that includes
    functionality to setup an MQTT connection, terminate it, maintain it, add subscribers,
    and add publishers.
    """

    MQTT_IP = "127.0.0.1"
    MQTT_PORT = 1883
    MQTT_TIMEOUT = 60
    REGISTRATION_TOPIC = "/registration"
    HEARTBEAT_TOPIC = "/heartbeat"
    HEARTBEAT_FREQUENCY = 10  # seconds

    def __init__(
        self: Any,
        mqtt_ip: str = MQTT_IP,
        mqtt_port: int = MQTT_PORT,
        mqtt_timeout: int = MQTT_TIMEOUT,
        registration_topic: str = REGISTRATION_TOPIC,
        heartbeat_topic: str = HEARTBEAT_TOPIC,
        heartbeat_frequency: int = HEARTBEAT_FREQUENCY,
    ) -> None:
        """BaseMQTTPubSub constructor takes constants for the config filepath, heartbeat channel,
        and heartbeat frequency and converts them to class accessible variables.
        Args:
            config_path (str, optional): relative file path to the config file that includes key
            value pairs for the ip address, port, and timeout (all config files should be stored
            in the ./config/ directory). Defaults to CONFIG_PATH.
            heartbeat_topic (str, optional): the topic name to pass a heartbeat payload to keep
            the TCP/IP connection with the MQTT broker alive (a default value is assigned because
            all container should publish a heartbeat to the same channel). Defaults
            to HEARTBEAT_TOPIC.
            heartbeat_frequency (int, optional): the frequency to publish the heartbeat at
            (currently unused but included for future development). Defaults to HEARTBEAT_FREQUENCY.
        """

        self.mqtt_ip = mqtt_ip
        self.mqtt_port = mqtt_port
        self.timeout = mqtt_timeout
        self.registration_topic = registration_topic
        self.heartbeat_topic = heartbeat_topic
        self.heartbeat_frequency = heartbeat_frequency

        self.connection_flag = None
        self.graceful_disconnect_flag = None

        self.client = mqtt.Client()

    def _parse_config(self: Any) -> Dict[str, str]:
        """Parses the config at the specified path from the constructor where the equal sign
        connects the key with the value and each pair is newline delimited (assumes utf-8 encoded).
        Returns:
            dict: a dictionary that maps the config keys to their values for programatic use.
        """
        with open(self.config_filepath, "r", encoding="utf-8") as file_pointer:
            parameters = {
                line.split("=")[0]: line.split("=")[-1]
                for line in file_pointer.read().splitlines()
            }
            return parameters

    def connect_client(self: Any) -> None:
        """Properly add a client connection to the MQTT server that includes a callback, which
        verifies that the connection was successful.
        """

        def _on_connect(
            _client: mqtt.Client,
            _userdata: Any,
            _flags: Dict[Any, Any],
            response_flag: int,
        ) -> None:
            """Connection callback that stores the result of connecting in a flag for class-wide
            access to verify the connection—can be overridden for more elaborate usage.
            Args:
                _client (mqtt.Client): the MQTT client that was instatntiated in the constructor.
                _userdata (Any):  data passed to the callback through the MQTT paho Client
                class contructor.
                _flags (dict):a dictionary that maps the response flags in the case that
                there are more than one.
                response_flag (int): integer response flag where 0 is success and 1 - 5
                are various failure types. The failure types can be found below and were taken
                from the MQTT paho docs:
                    0: Connection successful.
                    1: Connection refused - incorrect protocol version.
                    2: Connection refused - invalid client identifier.
                    3: Connection refused - server unavailable.
                    4: Connection refused - bad username or password.
                    5: Connection refused - not authorised 6-255: Currently unused.
            """
            if response_flag == mqtt.MQTT_ERR_SUCCESS:
                self.connection_flag = True
            else:
                self.connection_flag = False

        self.client.on_connect = _on_connect  # specify connection callback
        # connect to MQTT
        self.client.connect(
            self.mqtt_ip,
            self.mqtt_port,
            self.timeout,
        )

        self.client.loop_start()  # start callback thread

    def graceful_stop(self: Any) -> None:
        """How to properly shutoff the MQTT client connection that includes a callback to signal
        if the disconnect was successful.
        """

        def _on_disconnect(
            _client: mqtt.Client, _userdata: Any, response_flag: int
        ) -> None:
            """Disconnect callback that stores the result of diconnecting in a flag for class-wide
            access to verify the disconect—can be overridden for more elaborate usage.
            Args:
                _client (mqtt.Client): the MQTT client that was instatntiated in the constructor.
                _userdata (Any): data passed to the callback through the MQTT paho Client
                class contructor or set later using user_data_set().
                response_flag (int): integer response flag where 0 is success and 1 - 5
                are various failure types.
            """
            if response_flag == mqtt.MQTT_ERR_SUCCESS:
                self.graceful_disconnect_flag = True
            else:
                self.graceful_disconnect_flag = False

        self.client.disconnect()  # disconnect client gracefully
        self.client.on_disconnect = _on_disconnect  # specify disconnect callback
        self.client.loop_stop()  # TODO: not sure if this is necessary

    def setup_ungraceful_disconnect_publish(
        self: Any,
        ungraceful_disconnect_topic: str,
        ungraceful_disconnect_payload: str,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """If the container unexpectedly fails withouth calling disconnect(), the payload
        is published to the specified topic with the specified quality of service.
        Args:
            ungraceful_disconnect_topic (str): the topic to publish a message to.
            ungraceful_disconnect_payload (str): the message to publish at an ungraceful disconnect.
            qos (int, optional): MQTT quality of service options 0, 1, or 2. Defaults to 0.
            retain (bool, optional): keep the last published message or not. Defaults to False.
        """
        self.client.will_set(
            ungraceful_disconnect_topic, ungraceful_disconnect_payload, qos, retain
        )  # TODO: this function is untested

    def add_subscribe_topic(
        self: Any,
        topic_name: str,
        callback_method: Callable[[mqtt.Client, Dict[Any, Any], Any], None],
        qos: int = 2,
    ) -> bool:
        """Adds a callback to the topic specified with the specified quality of service.
        Args:
            topic_name (str): topic name to subscribe to.
            callback_method (Callable[[mqtt.Client, Dict[Any, Any], Any], None]): callback function to return information.
            qos (int, optional): MQTT quality of service options 0, 1, or 2. Defaults to 2.
        Returns:
            bool: returns True if adding callback was successful, else False.
        """
        self.client.message_callback_add(topic_name, callback_method)
        (result, _mid) = self.client.subscribe(topic_name, qos)
        return result == mqtt.MQTT_ERR_SUCCESS  # returns True if successful

    def add_subscribe_topics(
        self: Any,
        topic_list: List[str],
        callback_method_list: List[Callable[[mqtt.Client, Dict[Any, Any], Any], None]],
        qos_list: List[int],
    ) -> bool:
        """Adds topics, callbacks, and quality of services from lists and adds
        callbacks that subscribe topics of interest and returns True if all
        succeed. All must be in order.
        Args:
            topic_list (list[str]): list of topics to subscribe to.
            callback_method_list (list[Callable[[mqtt.Client, Dict[Any, Any], Any], None]]): list of callback functions to recieve callbacks.
            qos_list (list[str]): list of integers that correspond to the QoS requirements.
        Returns:
            bool: returns True if adding all callbacks was successful, else False.
        """
        result_list = []

        for idx, _val in enumerate(topic_list):
            self.client.message_callback_add(topic_list[idx], callback_method_list[idx])
            (result, _mid) = self.client.subscribe(topic_list[idx], qos_list[idx])
            result_list.append(result)
        return result_list == [mqtt.MQTT_ERR_SUCCESS] * len(
            topic_list
        )  # returns True if all successful

    def remove_subscribe_topic(self: Any, topic_name: str) -> None:
        """A wrapper around paho MQTT callback removal funciton, which does not send a
        success message so nothing is returned (TODO: is make a PR on the paho GitHub).
        Args:
            topic_name (str): topic string to remove subscriber callback from.
        """
        self.client.message_callback_remove(topic_name)

    def publish_to_topic(
        self: Any,
        topic_name: str,
        publish_payload: str,
        qos: int = 2,
        retain: bool = False,
    ) -> bool:
        """A wrapper around the paho MQTT publishing function publshes a payload to
        a topic name and returns True if successful.
        Args:
            topic_name (str): the topic name to publish to.
            publish_payload (str): string to publish to the topic.
            qos (int, optional): MQTT quality of service options 0, 1, or 2. Defaults to 2.
            retain (bool, optional): keep the last published message or not. Defaults to False.
        Returns:
            bool: returns true if publish succeded, else false
        """
        (result, _mid) = self.client.publish(topic_name, publish_payload, qos, retain)
        return result == mqtt.MQTT_ERR_SUCCESS  # returns True if successful

    def publish_registration(self: Any, payload: str) -> bool:
        """A function that includes an registration publisher. This is called in the
        constructor of a new node to broadcast its successful connection to MQTT.
        Args:
            payload (str): registration message to publish on initalization of a new node.
        Returns:
            bool: returns true if publish succeded, else false.
        """
        success = self.publish_to_topic(self.registration_topic, payload)
        return success

    def publish_heartbeat(self: Any, payload: str) -> bool:
        """A function that includes a hearbeat publisher. To call this function correctly,
        you will need to use the python schedule module around this function or put this in
        your main loop with a tick of self.heartbeat_frequency b/c MQTT is single threaded.
        Args:
            payload (str): heartbeat message to publish to keep TCP/IP connection alive.
        Returns:
            bool: returns true if publish succeded, else false.
        """
        success = self.publish_to_topic(self.heartbeat_topic, payload)
        return success

    def generate_payload_json(
        self: Any,
        push_timestamp: int,
        device_type: str,
        id_: str,
        deployment_id: str,
        current_location: str,
        status: str,
        message_type: str,
        model_version: str,
        firmware_version: str,
        data_payload_type: str,
        data_payload: str,
    ) -> str:
        """This function takes the requried parameters and creates a formatted data payload with
        headers for saving/database ingestion.
        Args:
            push_timestamp (int): The timestamp at which the message was pushed from device.
            device_type (str): This can be either 'Collector', 'Detector', 'Multimodal', etc.
            id_ (str): ID of the device. This could be IP address. This should remain constant.
            deployment_id (str): Device Deployment ID <Project>-<City>-ID.
            current_location (str): Can be set to null/blank if no GPS present.
            Attribute is required.
            status (str): Values can be - [Active, Deactive, Debug].
            message_type (str): Values can be - [Heartbeat, Event].
            model_version (str): Version string representing the model running on device.
            firmware_version (str): Device firmware version.
            data_payload_type (str): The type of payload to save
            e.g. [AIS, Telemetry, AudioFileName]
            data_payload (str): JSON string of data payload
            to publish.
        Returns:
            str: Returns a formatted JSON to publish.
        """
        out_json = {
            "PushTimestamp": push_timestamp,
            "DeviceType": device_type,
            "ID": id_,
            "DeploymentID": deployment_id,
            "CurrentLocation": current_location,
            "Status": status,
            "MessageType": message_type,
            "ModelVersion": model_version,
            "FirmwareVersion": firmware_version,
            data_payload_type: data_payload,
        }
        return json.dumps(out_json)