import { Header } from "@/components/header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-gray-950 dark:to-gray-900">
      <Header />
      <div className="container px-4 py-8 mx-auto max-w-7xl">
        <h1 className="mb-8 text-3xl font-bold text-center md:text-4xl">About Personalized Nutrition</h1>

        <div className="grid gap-8 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
              <CardDescription>Personalized nutrition for optimal gut health</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                The Personalized Nutrition Dashboard is designed to provide tailored dietary recommendations based on
                your unique health profile, symptoms, and dietary preferences.
              </p>
              <p>
                Our mission is to help individuals optimize their gut health through evidence-based nutrition
                recommendations that promote butyrate production and reduce inflammation.
              </p>
              <p>
                By combining advanced machine learning with nutritional science, we create personalized meal plans that
                address your specific health needs and goals.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>The Science of Butyrate</CardTitle>
              <CardDescription>Why butyrate matters for your gut health</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Butyrate is a short-chain fatty acid produced when beneficial gut bacteria ferment dietary fiber in your
                colon. It serves as the primary energy source for colon cells and plays a crucial role in gut health.
              </p>
              <p>Key benefits of butyrate include:</p>
              <ul className="pl-5 space-y-2 list-disc">
                <li>Strengthening the gut barrier</li>
                <li>Reducing inflammation</li>
                <li>Supporting immune function</li>
                <li>Improving insulin sensitivity</li>
                <li>Protecting against colorectal cancer</li>
              </ul>
              <p>
                Our meal plans are specifically designed to optimize butyrate production through carefully selected
                foods rich in prebiotic fibers and resistant starch.
              </p>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Our Technology</CardTitle>
              <CardDescription>Advanced AI for personalized nutrition</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                The Personalized Nutrition API uses XGBoost, a powerful machine learning algorithm, to analyze your
                health data and generate personalized recommendations.
              </p>
              <p>Key features of our technology:</p>
              <ul className="pl-5 space-y-2 list-disc">
                <li>
                  <strong>Digital Twin Simulation:</strong> Creates a virtual model of your gut microbiome to predict
                  how different foods will affect your health metrics.
                </li>
                <li>
                  <strong>Personalized Meal Planning:</strong> Generates meal plans tailored to your specific health
                  needs, symptoms, and dietary preferences.
                </li>
                <li>
                  <strong>Butyrate Optimization:</strong> Recommends foods that maximize butyrate production based on
                  your unique gut microbiome profile.
                </li>
                <li>
                  <strong>Inflammation Reduction:</strong> Identifies foods that may contribute to inflammation and
                  suggests alternatives to reduce symptoms.
                </li>
              </ul>
              <p>
                Our algorithm continuously learns and improves as more data becomes available, ensuring that our
                recommendations remain at the cutting edge of nutritional science.
              </p>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Contact Us</CardTitle>
              <CardDescription>Get in touch with our team</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <h3 className="mb-2 text-lg font-medium">Email</h3>
                  <p className="text-gray-600 dark:text-gray-300">support@personalizednutrition.example.com</p>
                </div>
                <div>
                  <h3 className="mb-2 text-lg font-medium">Address</h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    123 Health Street
                    <br />
                    Nutrition City, NC 12345
                    <br />
                    United States
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
}
