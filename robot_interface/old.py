import socket
import time

def start_server(host='0.0.0.0', port=3555):
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the address and port
    server_socket.bind((host, port))
    
    # Listen for incoming connections (backlog of 1)
    server_socket.listen(1)
    print(f"Server running on {host}:{port} - waiting for connection...")
    
    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print(f"Connected by {client_address}")

    #first ok
    data = client_socket.recv(1024)
    if not data:
        print("Client disconnected")

    message = data.decode('utf-8')
    print(f"Received from client: {message}")
    i = 1
    while True:

        print(f'cycle: {i}')
              
        # Send to the client
        response = str(i)
        client_socket.send(response.encode('utf-8'))

        # Receive data from the client
        data = client_socket.recv(1024)
        if not data:
            print("Client disconnected")
            break
        
        message = data.decode('utf-8')
        print(f"Received from client: {message}")

        #start mesure...
        time.sleep(5)

        # Send to the client
        response = '9'
        client_socket.send(response.encode('utf-8'))

        # Receive data from the client
        data = client_socket.recv(1024)
        if not data:
            print("Client disconnected")
            break
        
        message = data.decode('utf-8')
        print(f"Received from client: {message}")
        
        #start mesure...
        time.sleep(5)

        # Send to the client
        response = '9'
        client_socket.send(response.encode('utf-8'))

        # Receive data from the client
        data = client_socket.recv(1024)
        if not data:
            print("Client disconnected")
            break
        
        message = data.decode('utf-8')
        print(f"Received from client: {message}")

        #.....
        
        i = i + 1
    # Close connections
    client_socket.close()
    server_socket.close()
    print('closed')

if __name__ == "__main__":
    start_server()
