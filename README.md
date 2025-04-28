# Part 2 of CareerCompas

This is a seperate repo which includes the design and working of custom LangGraph workflow for the career Guidance project.
This was made for a pre-test of LangGraph before its commited to Main Project

## LangGraph Block diagram

[![photo-2025-04-23-08-58-54.jpg](https://i.postimg.cc/3RxhfPM6/photo-2025-04-23-08-58-54.jpg)](https://postimg.cc/mcJKtpgN)

## Detailed Project Information

Full project Report : https://drive.google.com/file/d/157zWNUAg2Iqiyok2beJmJmnmE90JosE0/view?usp=sharing 
(Don't worry its not some lame ass report it actually has some decent content)

## Screenshots

[![photo-2025-04-23-09-36-15.jpg](https://i.postimg.cc/rsSKMBmJ/photo-2025-04-23-09-36-15.jpg)](https://postimg.cc/xN10PFLJ)

## Run

To run the Frontend: ``` npx live-server frontend-chat ```
To run the LangGraph: ``` python main_final.py ``` 

## Troubleshooting

Here the system works completely fine, until it comes to the point where it is not able to conclude itself and instead it keeps on going and going further into all the interests. It seems that the LLM is not able to sort out positive and negative sentiments from the user prompt and this it is filling all empty. Thus it is not able to evaluate the list of career options and thus the chat keeps on continuing even after putting a conversation number limit. 
 
Probable solution: 
1) This issue can be solved but it needs heavy expertise in Python, building agents, and LangGraph. 
2) Using a different and better LLM to smartly understand the prompt embedded in the code. 

## Point to Note

Here the given block diagram is for main_final.py 
main_new.py, main_no_tags.py are not in working state.

main.py is a working state but it does not apply the given block diagram logic and uses another graph.
