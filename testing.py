import time
import debugpy

debugpy.listen(("0.0.0.0", 5678))  # Listen once, anywhere before point of interest
iter = 0
while True:
    time.sleep(1)
    if debugpy.is_client_connected():  # Only break if attached
        debugpy.breakpoint()  # Acts like a regular breakpoint in IDE
    print(iter)
    iter = iter + 1


# {
#     "name": "Python: Attach",
#     "type": "python",
#     "request": "attach",
#     "connect": {
#         "host": "localhost",
#         "port": 5678
#     },
#     "justMyCode": false
# }
