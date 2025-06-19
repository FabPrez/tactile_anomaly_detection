import socket

class RobotSocketInterface:
    def __init__(self, host='192.168.0.153', port=3555):
        self.host = host
        self.port = port
        self.client_socket = None
        self.server_socket = None
        self.connect()
        print(f"[RobotSocketInterface] Initialized on {host}:{port}")

    def connect(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[RobotSocketInterface] Waiting for robot connection...")
        self.client_socket, client_address = self.server_socket.accept()
        print(f"[RobotSocketInterface] Connected by {client_address}")

    def receive_message(self):
        data = self.client_socket.recv(1024)
        if not data:
            print("[RobotSocketInterface] Client disconnected")
            return None
        message = data.decode('utf-8').strip()
        print(f"[RobotSocketInterface] Received: {message}")
        return message

    def send_message(self, message):
        print(f"[RobotSocketInterface] Sending: {message}")
        self.client_socket.send(message.encode('utf-8'))

    def wait_for_message(self, expected_value):
        while True:
            msg = self.receive_message()
            if msg is None:
                return False
            if msg == expected_value:
                return True
            
    def select_pezzo(self, i):
        """
        Invia al robot il comando per selezionare il sottoprogramma del pezzo i.
        Esempio: i = 1 -> invia '1' al robot
        """
        message = str(i)
        self.send_message(message)
        print(f"[RobotSocketInterface] Selezionato pezzo {i}")

    def close(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[RobotSocketInterface] Connection closed")

