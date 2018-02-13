<center><img src="https://storage.googleapis.com/api.octadero.com/rada/1.gif">
</br><a href="https://www.octadero.com/2018/02/12/using-autoencoder-for-clusetring-political-events">Origin article</a></center></br>
<span style="font-weight: 400;">Machine learning algorithms have been put to good use in various areas for several years already. Analysis of various political events can become one of such areas. For instance, it can be used for predicting voting results, developing mechanisms for clustering the decisions made, analysis of political actors' actions. In this article, I will try to describe the result of a research in this area.</span>
</br>
<h4><b>Problem Definition  </b></h4>
<span style="font-weight: 400;">Modern machine learning capabilities allow converting and visualizing huge amounts of data. Thereby it became possible to analyze political parties' activities by converting voting instances that took place during 4 years into a self-organizing space of points that reflects actions of each elected official.</span>
<span style="font-weight: 400;">Each politician expressed themselves via 12 000 voting instances. Each voting instance can represent one of five possible actions (the person was absent, skipped the voting, voted approval, voted in negative, abstained).</span>
<span style="font-weight: 400;">The task is to convert the results of all voting instances into a point in the 3D Euclidean space that will reflect some considered attitude.</span>
</br>
<h4><b>Open Data</b></h4>
<span style="font-weight: 400;">The original data was taken from the <a href="http://data.rada.gov.ua/open">official website</a> and converted into <a href="https://github.com/Octadero/rada/tree/master/OpenData">intermediate data</a> for a neural network.</span>
</br>
<h4><b>Autoencoder</b></h4>
<span style="font-weight: 400;">Considering the problem definition, it is necessary to represent 12 000 voting instances as a vector of the 2 or 3 dimension. Humans can operate 2- or 3-dimension spaces, and it is quite difficult to imagine more spaces.</span>

<span style="font-weight: 400;">Let's apply autoencoder to decrease the capacity.</span>

<center><img class="" src="https://storage.googleapis.com/api.octadero.com/rada/image-autoencoder-net%402x.png" alt="" width="580" height="580" /> Simple Auto-encoder.</<center>>

<span style="font-weight: 400;">The autoencoder is based on two functions:</span>

<span style="font-weight: 400;"><img src="http://latex.codecogs.com/gif.latex?h%20%3D%20e%5Cleft%28x%20%5Cright%29">  - encoding function;</span>

<span style="font-weight: 400;"><img src="http://latex.codecogs.com/gif.latex?x%27%20%3D%20d%28h%29"> - decoding function;</span>

<span style="font-weight: 400;">The initial vector <img src="http://latex.codecogs.com/gif.latex?x"> with dimension <img src="http://latex.codecogs.com/gif.latex?m"> is supplied to the neural network as an input, and the network converts it into the value of the hidden layer <img src="http://latex.codecogs.com/gif.latex?h"> with dimension <img src="http://latex.codecogs.com/gif.latex?n">. After that the neural network decoder converts the value  of the hidden layer <img src="http://latex.codecogs.com/gif.latex?h"> into an output vector <img src="http://latex.codecogs.com/gif.latex?x"> with dimension <img src="http://latex.codecogs.com/gif.latex?m">, while <img src="http://latex.codecogs.com/gif.latex?m%20%3E%20n">.  That is, in the result the hidden layer <img src="http://latex.codecogs.com/gif.latex?h"> will be of lesser dimension, while being able to display all the range of the initial data.</span>

<span style="font-weight: 400;">Objective cost function is used for exercising the network:</span>
<p style="text-align: center;"><span style="font-weight: 400;"><img src="http://latex.codecogs.com/gif.latex?L%3D%28x%2C%20x%27%29%3D%28x%2C%20d%28e%28x%29%29"></span></p>
<span style="font-weight: 400;">In other words, the difference between the values of the input and output layers is minimized. Exercised neural network allows compressing the dimension of the initial data to some dimension <img src="http://latex.codecogs.com/gif.latex?n"> on the hidden layer <img src="http://latex.codecogs.com/gif.latex?h"> .</span>

<span style="font-weight: 400;">On the figure, you can see one input layer, one hidden layer and one output layer. There can be more such layers in a real-case scenario.</span>

<span style="font-weight: 400;">Now we are finished with the theoretical part, let's do some practice.</span>

<span style="font-weight: 400;">The data has been collected from the official site in the JSON format, and encoded into a vector already.</span>

<img class="" src="https://storage.googleapis.com/api.octadero.com/rada/image-vector%402x.png" alt="Input data encoding to vector." width="580" height="580" /> Input data encoding to vector.

<span style="font-weight: 400;">Now there is a dataset with dimension 24000 x 453. Let's create a neural network using the TensorFlow means:</span>
<div class="mceTemp"></div>

<span style="font-weight: 400;">The network will be exercised by the RMSProb optimizer with learning rate 0.01. </span><span style="font-weight: 400;">In the result, you can see the TensorFlow operation chart:</span>

<img class="" src="https://storage.googleapis.com/api.octadero.com/rada/image-Autoencoder_graph%402x.png" alt="Autoencoder TensorFlow Graph." width="580" height="561" /> Autoencoder TensorFlow Graph.

<span style="font-weight: 400;">For extra testing purposes, let's select the first four vectors and render their values as images on the neural network input and output. This way you can ensure that the values of the input and output layers are "identical" (to a tolerance).</span>

<center><img src="https://storage.googleapis.com/api.octadero.com/rada/canvas_orig.png" alt="Initial input data" width="400" height="400" /> Initial input data</center>

<img src="https://storage.googleapis.com/api.octadero.com/rada/canvas_recon.png" alt="Values of the neural network output layer" width="400" height="400" /> Values of the neural network output layer

<span style="font-weight: 400;">Now let's gradually pass all input data to the neural network and extract values of the hidden layer. These values are the compressed data in question. </span><span style="font-weight: 400;">Besides, I tried to select different layers and chose the configuration that allowed coming around minimum error. Origin is the diagram of the benchmark exercising.</span>
<h4><b>PCA and t-SNE dimension reducers</b></h4>
<span style="font-weight: 400;">On this stage, you have 450 vectors with dimension 128. This result is quite good, but it is not good enough to give it away to a human. That's why let's go deeper. Let's use the PCA and t-SNE approaches to lessen the dimension. There are many articles devoted to the principal component analysis method (</span><i><span style="font-weight: 400;">PCA</span></i><span style="font-weight: 400;">), so I won't include any descriptions herein, however, I would like to tell you about the</span><a href="https://distill.pub/2016/misread-tsne/"> <span style="font-weight: 400;">t-SNE</span></a><span style="font-weight: 400;"> approach. </span><span style="font-weight: 400;">The initial document,</span><a href="http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf"> <b>Visualizing data using t-SNE</b></a><span style="font-weight: 400;">, contains a detailed description of the algorithm; I will take reducing two-dimensional space to one-dimensional space as an example.</span>

<span style="font-weight: 400;">There is a 2D space and three classes (A, B, and C) located within this space. Let's try to project the classes to one of the axes.</span>

<img class="" src="https://storage.googleapis.com/api.octadero.com/rada/image-classes%402x.png" width="581" height="581" /> Classes

<img class="" src="https://storage.googleapis.com/api.octadero.com/rada/image-classes-projection%402x.png" width="580" height="580" /> Projection classes.

<span style="font-weight: 400;">As you can see, none of the axes is able to give us the broad picture of the initial classes. The classes get all mixed up, and, as a result, lose their initial characteristics. </span><span style="font-weight: 400;">The task is to arrange the elements in the eventual space maintaining the distance ratio they had in the initial space. That is, the elements that were close to each other should remain closer than those located farther. </span>
<h4><b>Stochastic Neighbor Embedding</b></h4>
<span style="font-weight: 400;">Let's convey the initial relation between the datapoints in the initial space as the distance between the points <img src="http://latex.codecogs.com/gif.latex?x_i">, <img src="http://latex.codecogs.com/gif.latex?x_j"> in Euclidean space: <img src="http://latex.codecogs.com/gif.latex?%5Cmathopen%7Cx_i%20-%20x_j%5Cmathclose%7C"> </span>and <img src="http://latex.codecogs.com/gif.latex?%5Cmathopen%7C%20y_i%20-%20y_j%20%5Cmathclose%7C"> correspondingly for the point in the space in question.

<span style="font-weight: 400;">Let's define conditional probabilities that represent similarities of points in the initial space:</span>
<p style="text-align: center;"><img src="http://latex.codecogs.com/gif.latex?p_%7Bij%7D%3D%5Cfrac%7Bexp%28-%20%5Cmathopen%7C%7Cx_i%20-%20x_j%5Cmathclose%7C%7C%20%5E2%20%2F2%5Csigma%5E2%29%7D%7B%20%5Csum_%7Bk%20%5Cneq%20l%7D%20exp%28-%20%5Cmathopen%7C%7Cx_k%20-%20x_l%5Cmathclose%7C%7C%20%5E2%20%2F2%5Csigma%5E2%29%7D"></p>
<span style="font-weight: 400;">This expression shows how close the point <img src="http://latex.codecogs.com/gif.latex?x_j"> is to <img src="http://latex.codecogs.com/gif.latex?x_i"> providing that you define the distance to the nearest datapoints in the class as Gaussian distribution centered at <img src="http://latex.codecogs.com/gif.latex?x_i"> with the given variance <img src="http://latex.codecogs.com/gif.latex?%5Csigma"> (centered at point <img src="http://latex.codecogs.com/gif.latex?x_i">).  Variance is unique for each datapoint and is determined separately based on the assumption that the points with higher density have lower variance.</span>

<span style="font-weight: 400;">Now let's describe the similarity of datapoint  and datapoint  correspondingly in the new space:</span>
<p style="text-align: center;"><img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D%3D%5Cfrac%7B%281%20%2B%20%5Cmathopen%7C%7Cy_i%20-%20y_j%5Cmathclose%7C%7C%20%5E2%29%5E%7B-1%7D%7D%7B%20%5Csum_%7Bk%20%5Cneq%20l%7D%281%20%2B%20%5Cmathopen%7C%7Cy_k%20-%20y_l%5Cmathclose%7C%7C%20%5E2%20%29%5E%7B-1%7D%7D"></p>
<span style="font-weight: 400;">Again, since we are only interested in modeling pairwise similarities, we set <img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D%20%3D%200">.</span>

<span style="font-weight: 400;">If the map points <img src="http://latex.codecogs.com/gif.latex?y_i"> and <img src="http://latex.codecogs.com/gif.latex?y_j"> correctly model the similarity between the high-dimensional datapoints <img src="http://latex.codecogs.com/gif.latex?x_i"> and <img src="http://latex.codecogs.com/gif.latex?x_j">, the conditional probabilities <img src="http://latex.codecogs.com/gif.latex?p_%7Bij%7D"> and <img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D"> will be equal. Motivated by this observation, SNE aims to find a low-dimensional data representation that minimizes the mismatch between <img src="http://latex.codecogs.com/gif.latex?p_%7Bij%7D"> and <img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D"> .</span>

<span style="font-weight: 400;">The algorithm finds the variance  for Gaussian distribution over each  datapoint <img src="http://latex.codecogs.com/gif.latex?x_i">. It is not likely that there is a single value of <img src="http://latex.codecogs.com/gif.latex?%5Csigma_i"> that is optimal for all datapoints in the data set because the density of the data is likely to vary. In dense regions, a smaller value of <img src="http://latex.codecogs.com/gif.latex?%5Csigma_i%20"> is usually more appropriate than in sparser regions. </span>

<span style="font-weight: 400;">SNE performs a binary search for the value of . </span><span style="font-weight: 400;">The search is performed considering a measure of the effective number of neighbors (perplexity parameter) that will be taken into account when calculating .</span>

<span style="font-weight: 400;">The authors of this algorithm found an example in physics, and describe the algorithm as a set of objects with various springs that are capable of repelling and attracting other objects. If the system is not interfered with for some time, it will find a stationary point by balancing the strain of all springs.</span>
<h4><b>t-Distributed Stochastic Neighbor Embedding</b></h4>
<span style="font-weight: 400;">The difference between the SNE and t-SNE algorithm is that t-SNE uses a Student-t distribution (also known as t-Distribution, t-Student distribution) rather than a Gaussian, and a symmetrized version of the SNE cost function.</span>

<span style="font-weight: 400;">That is, at first the algorithm locates all initial objects in the lower-dimensional space. After that it moves object by object basing on the distance between them (which objects were located closer/farther) in the initial space.</span>

<img class=" aligncenter" src="https://storage.googleapis.com/api.octadero.com/rada/image-classes-t-SNE%402x.png" width="800" height="800" />
<h4><b>TensorFlow, TensorBoard, and Projector.</b></h4>
<span style="font-weight: 400;">There is no need to implement such algorithms yourself nowadays. You can use such ready-to-use mathematical packages as scikit, MATLAB, or TensorFlow.</span>

<span style="font-weight: 400;"><a href="https://www.octadero.com/2017/12/01/visualizing-neural-network-exercising-with-means-of-tensorflowkit/">In my previous article, I mentioned that the TensorFlow toolkit contains a package for data and exercising process visualization called TensorBoard</a>. Let's use this solution. </span>


<img src="https://storage.googleapis.com/api.octadero.com/rada/1.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/2.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/3.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/4.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/5.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/6.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/7.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/8.gif">

<img src="https://storage.googleapis.com/api.octadero.com/rada/9.gif">

<span style="font-weight: 400;">There is another way, an entire portal called</span> <a href="http://octadero.com/event.php?p=http%3A%2F%2Fprojector.tensorflow.org"><span style="font-weight: 400;">projector</span></a><span style="font-weight: 400;"> that allows you to visualize your dataset directly on the Google server:</span>
<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Open the <a href="http://octadero.com/event.php?p=http%3A%2F%2Fprojector.tensorflow.org">TensorBoard Projector website</a>.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Click </span><b>Load Data.</b></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Select our dataset with vectors.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Add the metadata prepared earlier: labels, classes, etc.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Enable color map by one of the available columns.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Optionally, add JSON *.config file and publish data for public view.</span></li>
</ol>
<span style="font-weight: 400;">Now you can send the link to your analyst.</span>

<span style="font-weight: 400;">Those interested in the subject domain may find useful viewing various slices, for example:</span>
<ul>
<li><span style="font-weight: 400;">Distribution of votes of politicians from different regions.</span></li>
<li><span style="font-weight: 400;">Voting accuracy of different parties.</span></li>
<li><span style="font-weight: 400;">Distribution of voting of politicians from one party.</span></li>
<li><span style="font-weight: 400;">Similarity of voting of politicians from different parties.  </span></li>
</ul>
<h4><b>Conclusions</b></h4>
<ul>
<li><span style="font-weight: 400;">Autoencoders represent a range of simple algorithms that give surprisingly quick and good convergence result. </span></li>
<li><span style="font-weight: 400;">Automatic clustering does not answer the question about the nature of the initial data and requires further analysis; however, it provides a quick and clear vector that allows you to start working with your data.</span></li>
<li><span style="font-weight: 400;">TensorFlow and TensorBoard are powerful and fast-evolving tools for machine learning that allow solving tasks of diverse complexity. </span></li>
</ul>
