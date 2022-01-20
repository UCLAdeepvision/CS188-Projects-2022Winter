# UCLA DLCV Course Project

## Instruction to run this site locally

1. Clone this repo and enter

```
git clone git@github.com:UCLAdeepvision/CS188-Projects-2022Winter.git
cd CS188-Projects-2022Winter
```
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

1. Create a folder with your team id under ```./assets/images/```, you will use this folder to store all the images in your project.

2. Copy the template at ```./_posts/2022-01-18-instruction-to-post.md``` and rename it with format "year-month-date-your-project-name.md" under ```./_posts/```, for example 2022-01-19-my-final-project-name.md

3. Check out the sample post we provide at https://ucladeepvision.github.io/ and their source code at https://github.com/UCLAdeepvision/UCLAdeepvision.github.io/tree/main/_posts as well as basic Markdown syntax at https://www.markdownguide.org/basic-syntax/

4. Start your work in your .md file, please do NOT edit any other file in this repo.

Once you save the .md file, jekyll will synchronize the site.

## Submission
