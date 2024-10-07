from fastapi import FastAPI
from database import news, comments

app = FastAPI()


def comments_count(current_id):
    count = 0
    for current_comment in comments:
        if current_id == current_comment['news_id']:
            count += 1
    return count

def comments_list(current_id):
    list_comments = []
    for current_comment in comments:
        if current_id == current_comment['news_id']:
            list_comments.append(current_comment)
    return list_comments







@app.get("/")
async def get_news():
    data = []
    for current in news:
        if current['deleted'] == 'false':
            correct_item = current.copy()
            correct_item['comment_count'] = comments_count(current['id'])
            data.append(correct_item)
    data.append('news_count = ' + str(len(data)))
    return {"news": data},








@app.get("/news/{id}")
async def get_news_id(id):
    try:
        comment_dict = {}
        for current in news:
            if str(current['id']) == id and current['deleted'] == 'false':
                comment_dict = current.copy()
                comment_dict['comments'] = comments_list(current['id'])
                comment_dict['comments_count'] = len(comment_dict['comments'])
                break
        if len(comment_dict) == 0:
            return 404
        return {"comments": comment_dict},
    except:
        return 404
