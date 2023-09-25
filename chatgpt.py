import csv
import json
import asyncio
import aiohttp

OPENAI_API_KEY = ...  # add your API key here

prompt_template = '''Based on several news articles from a certain day (given in JSON format below, taken from that day's top headlines), use a number to indicate the overall impact on that day's Dow Jones Industrial Average (DJIA) index (if the news articles suggest that the index is expected to rise, give 1, if it is expected to fall, give 0. DO NOT give other values, such as "neutral").

%s

Output in this JSON format:

{"id": ..., "analysis": ..., "impact": ...}

Parameters:

- `id` (`int`): Same as the given `id` value.
- `analysis` (`str`): An analysis no more than 200 words. You must analyse before you reach a conclusion. You must not draw conclusions at the beginning.
- `impact` (`int`): The overall impact suggested by the news. This value can only be 1 or 0. If the news articles suggest that the index is expected to rise, give 1, if it is expected to fall, give 0. This value can only be 1 (representing a rise) or 0 (representing a fall). This is because providing an uncertain judgment would be meaningless. Regardless of whether stocks rise or fall, even a small fluctuation can have a significant impact due to leverage. You are not held responsible for your judgment results, but you must provide a reasonable and accurate analysis for your 1 (rise) or 0 (fall) judgment. You must consider only news that has a significant impact on the stock market and disregard other news and other irrelevant information.

Marking scheme:

- Always give a clear conclusion (rise/fall). If your `analysis` leads to an unclear conclusion, you will receive no point;
- If your `impact` value (0/1) is correct, you will receive full points (1000);
- If your `impact` value (0/1) is incorrect, you will be given 0-20 points based on your `analysis`;
- If your `impact` value is not 0 or 1, you will receive no point (therefore, you must not give any value other than 0 or 1);
- If your response is not a single JSON object, you will receive no point.'''

log_file = open('log.app', 'w')

def log(s):
    print(s)
    print(s, file=log_file)

async def ask(semaphore, prompt, label):
    while True:
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + OPENAI_API_KEY,
        }
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {"role": "system", "content": prompt},
            ],
            'temperature': 0,
        }

        try:
            log(f'Sending prompt: {prompt}')
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                        response_json = await response.json()
                        reply = response_json['choices'][0]['message']['content']
                        log(f'Got reply: {reply}, Real label: {label}')
                        return reply
        except Exception as e:
            log(f'Got error: {e}')
            pass

# Warning: the function uses `eval` and must not be evaluated on arbitrary data!
def sanitise_text(s: str) -> str:
    try:
        s = eval(s).decode('utf-8')
    except:
        pass
    return s.strip()

def parse_response(reply):
    try:
        obj = json.loads(reply)
        result = obj['impact']
        assert result in (0, 1)
    except Exception as e:
        log(f'Encountered error {e}')
        result = 1
    return result

def load_data():
    texts = []
    labels = []

    with open('Combined_News_DJIA.csv', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)

        for _, label, *news in reader:
            news = list(map(sanitise_text, news))
            label = int(label)
            texts.append(news)
            labels.append(label)

    len_total = len(texts)
    cutoff_train = int(len_total * 0.8)
    cutoff_eval = int(len_total * 0.9)

    data_train = texts[:cutoff_train]
    labels_train = labels[:cutoff_train]

    data_eval = texts[cutoff_train:cutoff_eval]
    labels_eval = labels[cutoff_train:cutoff_eval]

    data_test = texts[cutoff_eval:]
    labels_test = labels[cutoff_eval:]

    return {
        'data_train': data_train,
        'labels_train': labels_train,
        'data_eval': data_eval,
        'labels_eval': labels_eval,
        'data_test': data_test,
        'labels_test': labels_test,
    }

async def main():
    dataset = load_data()
    data_len = len(dataset['data_test'])

    semaphore = asyncio.Semaphore(10)
    replies = []
    for i, (news, label) in enumerate(zip(dataset['data_test'], dataset['labels_test']), 1):
        # if i <= 162:
        #     continue
        news_obj = json.dumps({
            "id": 1,
            "news": news,
        }, ensure_ascii=False, indent=2)
        prompt = prompt_template % (news_obj,)
        reply = asyncio.create_task(ask(semaphore, prompt, label))
        replies.append(reply)

    replies_ = await asyncio.gather(*replies)

    for reply, label in zip(replies_, dataset['labels_test']):
        result = parse_response(reply)

        log(f'{i}/{data_len}: Expected {label}, got {result}')

if __name__ == '__main__':
    asyncio.run(main())
