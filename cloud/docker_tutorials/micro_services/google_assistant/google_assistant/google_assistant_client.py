import re
from google_assistant import textinput
from logger import logger


def request_google_assistant(query, user_nm):
    logger.info("request_google_assistant")
    results = []
    res_message = textinput.get_response(query)
    if res_message is not None and res_message != 'error':
        # URLをリンクに
        pattern = "https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"
        urls = re.findall(pattern, res_message)
        for url in urls:
            res_message = res_message.replace(url, '<a href="' + url + '" target="_blank">' + url + '</a>')
        # ニックネーム変換
        if user_nm is not None:
            r = re.compile(r'chatnickname', re.IGNORECASE)
            res_message = re.sub(r, user_nm, res_message)
        res_message = res_message.replace('\\n', '<br>')
        results.append(res_message)

    return results
