package som

import Math._

object SelfOrgMap {
  case class Coord(x: Int, y: Int)
  case class Weight(r: Double, g: Double, b: Double) // Represents the color vector
  case class Node(coord: Coord, weight: Weight) // The node in the lattice

  object SOMUtils {
    
    private val rnd = new java.security.SecureRandom()
    def rndWeight = Weight(rnd.nextDouble(), rnd.nextDouble(), rnd.nextDouble())
    def squa(d: Double) = d * d
    def euclDist(w1: Weight, w2: Weight) = sqrt(squa(w1.r - w2.r) + squa(w1.g - w2.g) + squa(w1.b - w2.b)) // Euclidian distance
    def euclDist(n1: Node, n2: Node) = sqrt(squa(n1.coord.x - n2.coord.x) + squa(n1.coord.y - n2.coord.y)) // Euclidian distance
    def rndElem[T](list: List[T]): T = list(rnd.nextInt(list.size))
  }

  import SOMUtils._

  class Lattice(val size: Int, val nodes: List[Node])
  object Lattice {
    def apply(size: Int) = new Lattice(size, (for (x <- 0 until size; y <- 0 until size) yield Node(Coord(x, y), rndWeight)).toList)
  }
  
  class SOM(val size: Int, val numIterations: Int, val trainingSet: List[Weight]) {
  val mapRadius = size / 2.0
  val timeConstant = numIterations / log(mapRadius)
  def neighbourhoodRadius(iter: Double) = mapRadius * exp(-iter/timeConstant)
  def bmu(input: Weight, lattice: Lattice): Node = {
    val sortedNodesByDist = lattice.nodes.sortBy(n => euclDist(input, n.weight))
    sortedNodesByDist(0)
  }
  def bmuNeighbours(radius: Double, bmu: Node, lattice: Lattice): (List[(Node, Double)], List[(Node, Double)]) =
    lattice.nodes.map(n => (n, euclDist(n, bmu))).partition(n => n._2 <= radius)
  def learningRate(iter: Double) = 0.072 * exp(-iter/numIterations) // decays over time
  def theta(d2bmu: Double, radius: Double) = exp(-squa(d2bmu)/(2.0*squa(radius))) // learning proportional to distance
  def adjust(input: Weight, weight: Weight, learningRate: Double, theta: Double): Weight = {
    def adjust(iW: Double, nW: Double) = nW + learningRate * theta * (iW - nW)
    Weight(adjust(input.r, weight.r), adjust(input.g, weight.g), adjust(input.b, weight.b))
  }

  def nextLattice(iter: Int, lattice: Lattice): Lattice = {
    val randomInput = rndElem(trainingSet)
    val bmuNode = bmu(randomInput, lattice)
    val radius = neighbourhoodRadius(iter)
    val allNodes = bmuNeighbours(radius, bmuNode, lattice)
    val lrate = learningRate(iter)
    val adjustedNodes = allNodes._1.par.map(t => {
      val tTheta = theta(t._2, radius)
      val nWeight = adjust(randomInput, t._1.weight, lrate, tTheta)
      Node(t._1.coord, nWeight)
    }).toList
    new Lattice(lattice.size, adjustedNodes ++ allNodes._2.map(t => t._1))
  }

  def compute {
    import scala.annotation.tailrec
    
    @tailrec
    def helper(iter: Int, lattice: Lattice): Lattice =
      if (iter >= numIterations) lattice else helper(iter+1, nextLattice(iter, lattice))

    val endLattice = helper(0, Lattice(size))
//    UIUtils.persist(endLattice, "lattice")
  }
}
}