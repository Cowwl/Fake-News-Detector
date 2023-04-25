import requests

data = {
    "title": "Man Arrested for Using a Flamethrower to Melt Snow",
    "body": "Local resident Todd Fox has been detained for reckless endangerment and illegal use of high-powered fire-breathing weaponry for attacking snow with his flamethrower. Fox reportedly became so fed up with the week-long blowing snow epidemic in his area that he decided to KILL IT WITH FIRE. The neighborhood was treated with quite a show last night as Fox unleashed an inferno upon the mountainous snow palace that was his front yard. Neighbors to his immediate right and left noticed a bright orange cloud and could hear what they thought was puff the magic dragon spewing mayhem all over hell, which prompted one of them to notify police.",
}
response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
