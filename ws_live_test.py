"""Optional live test of the WebSocket endpoint against a REAL running server.
Terminal 1:  uvicorn app:app --port 8000
Terminal 2:  python ws_live_test.py
Client 1 drops while its reply is still being computed, client 3 leaves between turns;
clients 2 and 4 must keep working, and the server log shows a 'disconnected' line per drop."""
import asyncio
import json

import websockets

URL = "ws://127.0.0.1:8000/ws/chat"


async def main():
    ws1 = await websockets.connect(URL)
    print("client 1 connected:", json.loads(await ws1.recv()))
    await ws1.send(json.dumps({"question": "What is the standard notice period when I join?"}))
    await ws1.close()                                        # dropped mid-request
    print("client 1 dropped while its reply was being computed")

    ws2 = await websockets.connect(URL)
    await ws2.recv()
    await ws2.send("How long is my data kept if I am not hired?")
    print("client 2 answer:", json.loads(await ws2.recv())["answer"][:90])

    ws3 = await websockets.connect(URL)
    await ws3.recv()
    await ws3.send("What is the status of APP-0002?")
    await ws3.recv()
    await ws3.close()                                        # left between turns
    print("client 3 left after one turn")
    await asyncio.sleep(1.0)

    ws4 = await websockets.connect(URL)
    await ws4.recv()
    await ws4.send("What is the status of APP-0010?")
    print("client 4 answer (server still healthy):", json.loads(await ws4.recv())["answer"][:90])
    await ws4.close()
    await ws2.close()


asyncio.run(main())