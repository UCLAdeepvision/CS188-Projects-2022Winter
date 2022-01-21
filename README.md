# UCLA DLCV Course Project

Project page: https://ucladeepvision.github.io/CS188-Projects-2022Winter/

## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version at least 3.0.0, check https://www.ruby-lang.org/en/documentation/installation/ for instruction.

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
bundler install
bundle add webrick
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS188-Projects-2022Winter/ by default), go to this address with your browser.

## Working on the project

1. Create a folder with your team id under ```./assets/images/your-teamid```, you will use this folder to store all the images in your project.

2. Copy the template at ```./_posts/team00-instruction-to-post.md``` and rename it with format "yourteamid-projectshortname.md" under ```./_posts/```, for example, **team01-object-detection.md**

3. Check out the sample post we provide at https://ucladeepvision.github.io/CS188-Projects-2022Winter/ and the source code at https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/team00-instruction-to-post.md as well as basic Markdown syntax at https://www.markdownguide.org/basic-syntax/

4. Start your work in your .md file. You may only edit the .md file you just copied and renamed, and add images to ```./assets/images/your-teamid```. Please do NOT change any other files in this repo.

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.

## Deadlines
You should make three pull requests before the following deadlines:

*    January 27: Each group should determine the topic and list the 3 most relevant papers and their code repo.
*    February 24: Each group should include technical details and algorithms/code
*    March 14: Finalize blog article, Colab demo, and recorded video
