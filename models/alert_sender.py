from dotenv import load_dotenv
import os
import msal
import requests
class AlertSender:
    def __init__(self, dict, email):
        """we receive a dict with keys with folium's number"""
        self.dict  = dict
        self.body = ""
        self.email = email
    def __generateBody__(self):
        body = "Resumen par el dia testing\n"
        for folio in self.dict.keys():
            body+=f"""Alertas para el folio {folio}:\n"""
            print(self.dict[folio])
            current_df = self.dict[folio]
            for index, row in current_df.iterrows():

                body+= f""" El dia {row['FechaPrecio']} se registró una variación del {row['Diff_Pct']} \n"""
            body+="------------------\n\n"
        self.body = body
    def send_alert(self):
        load_dotenv()
        self.__generateBody__()
        subject = "Envío de alertas"
        TENANT_ID = os.getenv('TENANT_ID')
        AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
        SCOPE     = ["https://graph.microsoft.com/.default"]
        CLIENT_ID = os.getenv('CLIENT_ID')
        CLIENT_SECRET = os.getenv('CLIENT_SECRET')
        USER_ID = os.getenv("USER_ID")
        
        app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET
        )
        result = app.acquire_token_for_client(scopes=SCOPE)
        if "access_token" not in result:
            err = result.get("error_description", "Error desconocido al obtener token")
            raise RuntimeError(f"MSAL error: {err}")

        token = result["access_token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # 2. Prepara el payload de Graph
        email_msg = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "Text",
                    "content": self.body
                },
                "toRecipients": [
                    {"emailAddress": {"address": self.email}}
                ]
            },
            "saveToSentItems": True
        }

        # 3. Llama al endpoint sendMail
        endpoint = f"https://graph.microsoft.com/v1.0/users/{USER_ID}/sendMail"
        response = requests.post(endpoint, headers=headers, json=email_msg)

        if response.status_code == 202:
            return f"Correo enviado correctamente"
        else:
            # Devuelve el error para facilitar debugging
            return (f"Error al enviar correo ({response.status_code}): "
                    f"{response.text}")        

