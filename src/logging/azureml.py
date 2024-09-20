import logging
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from dotenv import load_dotenv

from src.utils import get_logger

log = get_logger(__name__)


def setup_azure_ml_client():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        log.debug(ex)
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        # This will open a browser page
        credential = InteractiveBrowserCredential()

    try:
        ml_client = MLClient.from_config(credential=credential)
    except Exception as ex:
        log.debug(ex)
        # use the environment variables in the dotfile instead.
        load_dotenv()
        subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
        resource_group = os.environ["AZURE_RESOURCE_GROUP"]
        workspace_name = os.environ["AZURE_WORKSPACE_NAME"]

        client_config = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
        }

        # write and reload from config file
        import json

        config_path = ".azureml/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            fo.write(json.dumps(client_config))
        ml_client = MLClient.from_config(credential=credential, path=config_path)

    log.info(ml_client)
    logger = logging.getLogger("azure")
    logger.setLevel(logging.WARNING)
    return ml_client
