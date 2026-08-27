#!/usr/bin/env python3
import asyncio
import json
import os
import uuid

import httpx

A2A_URL = os.environ.get("A2A_URL", "http://127.0.0.1:9016/a2a/")


def _build_message_payload(question: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "parts": [{"kind": "text", "text": question}],
                "messageId": str(uuid.uuid4()),
            }
        },
        "id": 1,
    }


def _build_poll_payload(task_id) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {"id": task_id},
        "id": 2,
    }


def _find_last_non_user_message(history: list) -> dict | None:
    for msg in reversed(history):
        if msg.get("role") != "user":
            return msg
    return None


def _print_task_history_summary(result: dict) -> None:
    if "history" not in result:
        return
    history = result["history"]
    if not history:
        return

    last_msg = _find_last_non_user_message(history)
    if last_msg and "parts" in last_msg:
        print("\n--- Agent Response ---")
        for part in last_msg["parts"]:
            if "text" in part:
                print("Agent response content omitted.")
            elif "content" in part:
                print("Agent response content omitted.")
    elif last_msg:
        print("Final response received without structured parts.")
    else:
        print("\n--- No Agent Response Found in History ---")


def _print_task_status_and_check_done(poll_data: dict) -> bool:
    """Print the task's current state; return True once it has reached a
    terminal (non in-progress) state."""
    state = poll_data["result"]["status"]["state"]
    print(f"Task State: {state}")
    if state in ("submitted", "running", "working"):
        return False

    print(f"\nTask Finished with state: {state}")
    _print_task_history_summary(poll_data["result"])
    print("Validation result received; body omitted.")
    return True


async def _poll_task(client: httpx.AsyncClient, url: str, task_id) -> None:
    while True:
        await asyncio.sleep(2)
        poll_payload = _build_poll_payload(task_id)
        poll_resp = await client.post(
            url, json=poll_payload, headers={"Content-Type": "application/json"}
        )

        if poll_resp.status_code != 200:
            print(f"Polling Failed: {poll_resp.status_code}")
            print(f"Polling failed with HTTP {poll_resp.status_code}.")
            return

        poll_data = poll_resp.json()
        if "result" not in poll_data:
            print("Starting polling error key check...")
            if "error" in poll_data:
                print(
                    f"Polling JSON-RPC error code: {poll_data['error'].get('code', 'unknown')}"
                )
            return

        if _print_task_status_and_check_done(poll_data):
            return


async def _handle_initial_response(client: httpx.AsyncClient, data: dict) -> None:
    if "result" in data and "id" in data["result"]:
        task_id = data["result"]["id"]
        print("\nTask submitted; polling for result...")
        await _poll_task(client, A2A_URL, task_id)

    if "error" in data:
        print(f"JSON-RPC error code: {data['error'].get('code', 'unknown')}")


async def _submit_question(client: httpx.AsyncClient, question: str) -> None:
    print("\nSubmitting the configured validation query.")
    print("--- Sending Request ---")

    payload = _build_message_payload(question)

    try:
        url = A2A_URL
        print("Trying the configured endpoint with JSON-RPC (message/send)...")
        resp = await client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.status_code}")
            print(f"Response body omitted (HTTP {resp.status_code}).")
            return

        try:
            data = resp.json()
            print("JSON response received.")
            await _handle_initial_response(client, data)
        except json.JSONDecodeError:
            print(f"Response body omitted (HTTP {resp.status_code}).")

    except httpx.RequestError as e:
        print(f"Operation failed: {type(e).__name__}")


async def main():
    print("Validating the configured A2A agent...")

    questions = [
        os.environ.get("A2A_VALIDATION_QUERY", "Describe your available capabilities.")
    ]

    async with httpx.AsyncClient(timeout=10000.0) as client:
        for q in questions:
            await _submit_question(client, q)


if __name__ == "__main__":
    asyncio.run(main())
