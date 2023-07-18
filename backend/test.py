import asyncio
import websockets

async def test():
    async with websockets.connect('wss://127.0.0.1:8000') as websocket:
        websocket.send("User: Hello.\nmedGPT:")
        while True:
            response = await websocket.recv()
            if response == "<EOA>":
                break
            print(response, end=" ")
asyncio.get_event_loop().run_until_complete(test())
