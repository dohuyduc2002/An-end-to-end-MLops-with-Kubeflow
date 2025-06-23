from pathlib import Path
from kfp import dsl
from component_utils import BASE_IMAGE, TARGET_IMAGE


@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE,
               packages_to_install=["slack_sdk"])
def slack_notification(
    slack_channel: str,
    message: str,
    slack_bot_token: str
):
    from slack_sdk import WebClient
    client = WebClient(token=slack_bot_token)
    client.chat_postMessage(
        channel=slack_channel,
        text=message
    )
