import numpy as np
from scapy.all import * # Используем rdpcap для чтения из файла
import sys
import os
from collections import OrderedDict
import argparse

def get_session_key(pkt):
    """Создает ключ сессии (5-tuple) в каноническом порядке."""
    # Проверяем наличие IP и TCP/UDP слоев
    if IP not in pkt or (TCP not in pkt and UDP not in pkt):
        return None

    try:
        proto_layer = pkt.getlayer(TCP) or pkt.getlayer(UDP)
        ip_layer = pkt.getlayer(IP)

        # Извлекаем 5-tuple
        proto = ip_layer.proto # 6 для TCP, 17 для UDP
        sport = proto_layer.sport
        dport = proto_layer.dport
        srcip = ip_layer.src
        dstip = ip_layer.dst

        # Приводим к каноническому виду (меньший IP/порт всегда первыми)
        # Это гарантирует, что оба направления потока попадут в одну сессию
        if (srcip, sport) > (dstip, dport):
            srcip, dstip = dstip, srcip
            sport, dport = dport, sport
        return (srcip, sport, dstip, dport, proto)
    except AttributeError:
        # Если у пакета нет нужных полей (например, фрагментированный без TCP/UDP заголовка)
        return None

def extract_first_session_all_layers(pcap_file, target_bytes=784):
    """
    Извлекает первые N байт данных (все слои, начиная с IP)
    из первой найденной TCP/UDP сессии в pcap файле.

    Args:
        pcap_file (str): Путь к pcap файлу.
        target_bytes (int): Желаемое количество байт.

    Returns:
        bytearray: Данные сессии (обрезанные/дополненные),
                   или None, если сессия не найдена или произошла ошибка.
    """
    first_session_key = None
    session_data = bytearray()
    bytes_collected = 0

    print(f"Чтение pcap файла: {pcap_file}...")
    try:
        # Используем PcapReader для потокового чтения, если файлы большие
        with PcapReader(pcap_file) as pcap_reader:
            for pkt_num, pkt in enumerate(pcap_reader):
                if bytes_collected >= target_bytes and first_session_key is not None:
                    break # Уже собрали достаточно для первой сессии

                key = get_session_key(pkt)
                if not key:
                    continue # Пропускаем пакеты без сессионного ключа

                # Если это первая сессия, которую мы встретили
                if first_session_key is None:
                    first_session_key = key
                    print(f"Найдена первая сессия: {first_session_key}")

                # Если пакет принадлежит нашей первой сессии
                if key == first_session_key:
                    # Извлекаем ВСЕ байты пакета, начиная с IP заголовка
                    # (или можно с Ethernet, если нужно)
                    if IP in pkt:
                        # Сериализуем слой IP и все, что после него
                        raw_packet_bytes = bytes(pkt.getlayer(IP))
                        bytes_to_add = min(len(raw_packet_bytes), target_bytes - bytes_collected)
                        if bytes_to_add > 0:
                            session_data.extend(raw_packet_bytes[:bytes_to_add])
                            bytes_collected += bytes_to_add
                        # print(f"  Добавлено {bytes_to_add} байт из пакета {pkt_num+1}, всего {bytes_collected}")


    except FileNotFoundError:
        print(f"Ошибка: Pcap файл не найден: {pcap_file}")
        return None
    except Scapy_Exception as e:
         print(f"Ошибка Scapy при чтении {pcap_file}: {e}")
         return None
    except Exception as e:
        print(f"Неизвестная ошибка при чтении {pcap_file}: {e}")
        return None

    if first_session_key is None:
        print("В pcap файле не найдено подходящих TCP/UDP сессий.")
        return None

    print(f"Извлечено {bytes_collected} байт из первой сессии.")

    # Дополнение нулями, если нужно
    if bytes_collected < target_bytes:
        padding_size = target_bytes - bytes_collected
        print(f"Дополнение {padding_size} нулями...")
        session_data.extend(b'\x00' * padding_size)
    # Обрезка не требуется, так как мы останавливаем сбор на target_bytes

    return session_data # Возвращаем bytearray

def prepare_input_file_simplified(pcap_file, output_file, n=784):
    """
    Извлекает данные первой сессии (All Layers) из pcap, подготавливает их
    (padding/truncation до N байт, normalization в float32)
    и сохраняет в бинарный файл.

    Args:
        pcap_file (str): Путь к исходному pcap файлу.
        output_file (str): Путь к выходному бинарному файлу (.bin).
        n (int): Целевое количество байт (и float'ов).
    """
    session_bytearray = extract_first_session_all_layers(pcap_file, target_bytes=n)

    if session_bytearray is None:
        print("Не удалось извлечь данные сессии.")
        return False

    # Преобразуем bytearray в numpy массив uint8
    prepared_bytes = np.frombuffer(session_bytearray, dtype=np.uint8)

    # Нормализация в float32
    print("Нормализация байт (деление на 255.0) в float32...")
    normalized_floats = prepared_bytes.astype(np.float32) / 255.0

    # Проверка размера перед сохранением
    if len(normalized_floats) != n:
         print(f"Ошибка: Неожиданный размер данных после подготовки: {len(normalized_floats)} (ожидалось {n})")
         return False

    # Сохранение в бинарный файл
    print(f"Сохранение {len(normalized_floats)} float32 значений в файл: {output_file}")
    try:
        with open(output_file, 'wb') as f:
            normalized_floats.tofile(f)
        print(f"Файл '{output_file}' успешно сохранен.")
        return True
    except Exception as e:
        print(f"Ошибка сохранения файла {output_file}: {e}")
        return False

# --- Основной блок для запуска из командной строки ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка входного бинарного файла (.bin) для модели NMDL из pcap файла.")
    parser.add_argument("pcap_input", help="Путь к входному pcap файлу.")
    parser.add_argument("binary_output", nargs='?', default="prepared_input.bin",
                        help="Путь к выходному бинарному файлу (по умолчанию: prepared_input.bin).")
    parser.add_argument("-n", "--num_bytes", type=int, default=784,
                        help="Целевое количество байт/float для входного тензора (по умолчанию: 784).")

    args = parser.parse_args()

    if not os.path.exists(args.pcap_input):
         print(f"Ошибка: Входной pcap файл не найден: {args.pcap_input}")
    else:
        prepare_input_file_simplified(args.pcap_input, args.binary_output, n=args.num_bytes)