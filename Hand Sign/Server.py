import socket

serverPort = 80
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.bind(('', serverPort))

serverSocket.listen(1)
connectionSocket, addr = serverSocket.accept()
count = 0
print('Server is ready to serve...')
while True:
    # print('Server is ready to serve...')
    message = connectionSocket.recv(1024).decode()
    if message == 'left':
        print('Car turn left')
    elif message == 'right':
        print('car turn right')
    elif message == 'forward':
        print('car move forward')
    elif message == 'backward':
        print('car move backward')
    else:
        print('car stop')
    if 0xFF == ord('q'):
        break
connectionSocket.close()
serverSocket.close()


