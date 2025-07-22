import { BarChart3, FileUp, ClipboardList, LineChart, ShieldCheck, Users } from "lucide-react"

export function FeatureSection() {
  const features = [
    {
      icon: <FileUp className="h-10 w-10 text-primary" />,
      title: "Upload Microbiome Data",
      description: "Easily upload your microbiome data files in various formats including CSV and TSV.",
    },
    {
      icon: <ClipboardList className="h-10 w-10 text-primary" />,
      title: "Track Symptoms",
      description: "Record your symptoms and health information to get more accurate insights.",
    },
    {
      icon: <BarChart3 className="h-10 w-10 text-primary" />,
      title: "Data Analysis",
      description: "Advanced analysis using machine learning techniques to identify patterns in your microbiome.",
    },
    {
      icon: <LineChart className="h-10 w-10 text-primary" />,
      title: "Visualize Results",
      description: "Interactive charts and visualizations to help understand your microbiome health.",
    },
    {
      icon: <Users className="h-10 w-10 text-primary" />,
      title: "Doctor Insights",
      description: "Receive personalized health recommendations from qualified healthcare professionals.",
    },
    {
      icon: <ShieldCheck className="h-10 w-10 text-primary" />,
      title: "Secure & Private",
      description: "Your data is encrypted and protected in compliance with health data privacy regulations.",
    },
  ]

  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-muted/50">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">Key Features</h2>
            <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Our platform provides comprehensive tools for both patients and healthcare providers
            </p>
          </div>
        </div>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 py-12 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <div key={index} className="flex flex-col items-center space-y-2 rounded-lg border p-6 shadow-sm">
              {feature.icon}
              <h3 className="text-xl font-bold">{feature.title}</h3>
              <p className="text-center text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
