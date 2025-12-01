Лабораторная работа №3
Задание:
"Реализовать программу для накладывания фильтров на изображения. Возможные фильтры: размытие, выделение границ, избавление от шума. Изменить время Для работы с графическими файлами рекомендуется использовать libpng (man libpng). Примеры использования библиотеки в /usr/share/doc/libpng12-dev/examples/ Считать изображение из файла, преобразовать в массив, отправить на CUDA Device, провести на устройстве обработку фильтром, вернуть изображение в память хоста, преобразовать в картинку, сохранить файл. Использовать 2 видеокарты"

Характеристики устройства
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-16GB           On  |   00000000:61:00.0 Off |                    0 |
| N/A   35C    P0             26W /  250W |     254MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-PCIE-16GB           On  |   00000000:DB:00.0 Off |                    0 |
| N/A   59C    P0             96W /  250W |    7057MiB /  16384MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+


Device name: Tesla V100-PCIE-16GB
Number of multiprocessors: 80
Global memory size: 16928342016 bytes
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64

Device name: Tesla V100-PCIE-16GB
Number of multiprocessors: 80
Global memory size: 16928342016 bytes
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
Результаты работы
Blur filter
Loaded image: test_img.png (1024 x 1536), channels = 4
GPU 0:
Host -> Device copy time: 0.940 ms
Kernel execution time: 0.077 ms
Device -> Host copy time: 1.654 ms
GPU 1:
Host -> Device copy time: 0.948 ms
Kernel execution time: 1.459 ms
Device -> Host copy time: 1.565 ms
Total GPU time (pipeline): 17.151 ms
Edge filter
GPU 0:Loaded image: test_img.png (1024 x 1536), channels = 4
GPU 0:
Host -> Device copy time: 0.937 ms
Kernel execution time: 0.078 ms
Device -> Host copy time: 2.051 ms
GPU 1:
Host -> Device copy time: 0.937 ms
Kernel execution time: 1.463 ms
Device -> Host copy time: 1.125 ms
Total GPU time (pipeline): 14.733 ms
Denoise filter
Loaded image: test_img.png (1024 x 1536), channels = 4
GPU 0:
Host -> Device copy time: 0.951 ms
Kernel execution time: 0.081 ms
Device -> Host copy time: 1.650 ms
GPU 1:
Host -> Device copy time: 0.918 ms
Kernel execution time: 1.491 ms
Device -> Host copy time: 1.536 ms
Total GPU time (pipeline): 16.420 ms
Вывод
Время
В ходе работы была реализована параллельная обработка изображения на двух видеокартах Tesla V100. Изображение делилось по горизонтали на две части, каждая из которых обрабатывалась на своём GPU.

Результаты показали, что:

Основное время выполнения приходится на копирование данных между оперативной памятью и устройством (H2D ≈ 0.9 ms, D2H ≈ 1.1–1.6 ms). Собственно вычисления фильтра занимают доли миллисекунды на GPU0 и ~1.45 ms на GPU1 (что объясняется более сложной загрузкой GPU1).

Все три фильтра (blur, edge, denoise) имеют близкие времена выполнения: ~14.7–17.1 ms на весь пайплайн.