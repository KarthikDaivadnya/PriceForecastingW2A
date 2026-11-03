
import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


FASTAPI_URL = "http://127.0.0.1:8000/predict"


@api_view(["POST"])
def generate_forecast(request):
    historical_prices = request.data.get("historical_prices")
    last_date = request.data.get("last_date")

    if not historical_prices or not last_date:
        return Response(
            {"error": "historical_prices and last_date are required"},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        response = requests.post(
            FASTAPI_URL,
            json={
                "historical_prices": historical_prices,
                "last_date": last_date
            },
            timeout=60
        )

        response.raise_for_status()

        return Response(
            response.json(),
            status=status.HTTP_200_OK
        )

    except requests.RequestException as error:
        return Response(
            {"error": "Prediction service unavailable", "details": str(error)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
