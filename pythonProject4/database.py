import json


string = {
  "news": [
    {
      "id": 1,
      "title": "Новость 1",
      "date": "2019-01-01T20:56:35",
      "body": "The news",
      "deleted": "false",
    },
    {
          "id": 2,
          "title": "Новость 2",
          "date": "2019-01-01T20:56:35",
          "body": "The news",
          "deleted": "false",
    },
    {
      "id": 3,
      "title": "Новость 3",
      "date": "2019-01-01T20:56:35",
      "body": "The news",
      "deleted": "true",
    },
  ],
}

json_string = json.dumps(string)
string = {
  "comments": [
    {
      "id": 1,
      "news_id": 1,
      "title": "Комментарий к первой новости",
      "date": "2024-01-02T21:58:25",
      "comment": "Comment",
    },
    {
      "id": 2,
      "news_id": 1,
      "title": "Комментарий к первой новости",
      "date": "2024-01-02T21:58:25",
      "comment": "Comment",
    },
{
      "id": 3,
      "news_id": 2,
      "title": "Комментарий ко второй новости",
      "date": "2024-01-02T21:58:25",
      "comment": "Comment",
    },
    {
      "id": 4,
      "news_id": 2,
      "title": "Комментарий ко второй новости",
      "date": "2024-01-02T21:58:25",
      "comment": "Comment",
    },
  ],

}

data_news = json.loads(json_string)
json_string = json.dumps(string)
data_comments = json.loads(json_string)

comments = []
for data in data_comments['comments']:
    comments.append(data)
news = []
for data in data_news["news"]:
    news.append(data)