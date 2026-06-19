"""
This class for sending the user report to his email address
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from dataclasses import dataclass

#_______________________________________
# Result Data class
#_______________________________________
@dataclass
class EmailResult:
  success : bool
  message : str
  error : str | None = None
#_______________________________________
# Sending Email Class
#_______________________________________
class EmailSender:
  """
  Sends physiognomy reports via Gmail SMTP.
  """
  SMTP_HOST = "smtp.gmail.com"
  SMTP_PORT = 587   
  def __init__(self):
    # Load credentials from HuggingFace Secrets
    self.gmail_address  = self._get_secret("GMAIL_ADDRESS")
    self.gmail_password = self._get_secret("GMAIL_APP_PASSWORD")
    if not self.gmail_address or not self.gmail_password:
      raise ValueError("Gmail credentials not found.")

  def _get_secret(self, key):
    return os.environ.get(key)
    
  def send_report(self, to_email, report_text, session_id) -> EmailResult:
    """
    Main Method to send the report email to the user 
    """
    #__1 Build MIME email text in html 
    try:
      html_body = f"""<html><body>
      <h2>Your Physiognomy Reading</h2>
      <p>Session ID: <code>{session_id}</code></p>
      <hr>
      <div style="font-family: Georgia; line-height: 1.6; max-width: 700px;">
      Ahln Ahln ^ر^ <br>
      {report_text.replace(chr(10), '<br>')}
      <p>Thank you for using the Face Physiognomy Analyzer</p>
      <p> Here is your personalized reading based on facial morphology analysis.</p>
      </div>
      <hr><p style="color: gray; font-size: 12px;">
      Note: This reading is based on the principles described in
      "Amazing Face Reading" by Mac Fulfer and is intended for educational and entertainment purposes only.</p>
      <p>  Best regards,<br>Yasmeen & Lolo ☆*: .｡. o(≧▽≦)o .｡.:*☆"</p>
      </body></html>
      """
      msg = MIMEText(html_body,  "html")
      msg["From"] = self.gmail_address
      msg["To"] = to_email
      msg["Subject"] = f"Your Physiognomy Reading - Session {session_id}"
      
      context = ssl.create_default_context()
      with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT) as server:   #__2 Connect to Gmail SMTP server
        server.ehlo()
        server.starttls(context=context)   #__3 Strat TLS encryption for security staff
        server.login(self.gmail_address,self.gmail_password)
        server.sendmail(self.gmail_address, to_email, msg.as_string())     #__4 Send the email & Disconeect
        return EmailResult(success = True, message = f"Report sent to {to_email}")
    except smtplib.SMTPAuthenticationError:
      return EmailResult(success=False, message="Authentication failed", error = "Gmail authentication failed")
    except Exception as e:
      return EmailResult(success=False, message="Email failed", error= str(e))



    
    

    
    





