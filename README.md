# Vision Testing Multi-Modal Models

Ever wonder how well GPT-4V does at recalling text from an image?

Let's try it out

Want to see an overview video? Check out [this tweet](https://twitter.com/GregKamradt/status/1773386891368521806).

```
git clone https://github.com/gkamradt/MultiModalVisionTesting

cd MultiModalVisionTesting

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 main.py
```

This repo already contains all of the results and images from the test.

If you want want to run a single test by yourself, delete the result from the results file you want to run and then run the script.

Or if you want to start completely from scratch, uncomment `vtt.reset_results()` in `main.py`


Shoutout to [Bryan Bischof](https://twitter.com/BEBischof) for his conversations and feedback on this project.

Note: This repo will likely move over to the [Needle/Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) repository as we build out more tests there

Contributions welcome, but help would be higher leverage by implmenting this on [needle/haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) first.

LICENSE: Feel free to do whatever you want with the code, attribution would be appreciated.

Made with ❤️ by [Greg Kamradt](https://twitter.com/GregKamradt)