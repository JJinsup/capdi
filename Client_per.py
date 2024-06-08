import socket
import time

def measure_performance(client_socket, number, num_packets=100):
    delays = []
    send_times = []
    recv_times = []
    lost_packets = 0

    for _ in range(num_packets):
        send_time = time.time()
        try:
            client_socket.send(str(number).encode())
            data = client_socket.recv(1024)
            recv_time = time.time()
            squared = int(data.decode())
            print(f"전송한 숫자: {number}, 받은 제곱값: {squared}")
            delays.append(recv_time - send_time)
            send_times.append(send_time)
            recv_times.append(recv_time)
        except socket.timeout:
            lost_packets += 1

    # Calculate performance metrics
    total_time = recv_times[-1] - send_times[0] if recv_times else 0
    bitrate = (num_packets - lost_packets) * 1024 * 8 / total_time if total_time > 0 else 0  # in bits per second
    avg_delay = sum(delays) / len(delays) if delays else 0
    jitter = sum(abs(delays[i] - delays[i-1]) for i in range(1, len(delays))) / (len(delays) - 1) if len(delays) > 1 else 0
    loss_rate = lost_packets / num_packets if num_packets > 0 else 0

    return avg_delay, bitrate, jitter, loss_rate

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 6418))
    
    while True:
        number = input("서버로 전송할 정수를 입력하세요 (종료하려면 'exit' 입력): ")
        if number.lower() == 'exit':
            break

        try:
            number = int(number)
        except ValueError:
            print("정수를 입력하세요.")
            continue

        avg_delay, bitrate, jitter, loss_rate = measure_performance(client_socket, number)
        print(f"Average Delay: {avg_delay:.6f} seconds")
        print(f"Bitrate: {bitrate:.2f} bps")
        print(f"Jitter: {jitter:.6f} seconds")
        print(f"Loss Rate: {loss_rate:.2%}")
    
    client_socket.close()
    print("클라이언트 연결 종료")

if __name__ == "__main__":
    main()
