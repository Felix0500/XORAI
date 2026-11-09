using System;
using System.Globalization;

class Program
{
    static void Main()
    {
        Random r = new Random();

        double[,] möglicheEingaben ={{ 0, 0 },{ 0, 1 },{ 1, 0 },{ 1, 1 }};
        double[] möglicheAusgaben = {0,1,1,0};

        Knoten[] eingabeSchicht = {new Knoten("e1"), new Knoten("e2")};
        Knoten[] versteckteSchicht = 
        {
            new Knoten("v1",[eingabeSchicht[0],eingabeSchicht[1]],[r.NextDouble(),r.NextDouble()]), 
            new Knoten("v2",[eingabeSchicht[0],eingabeSchicht[1]],[r.NextDouble(),r.NextDouble()]), 
        };

        Knoten[] ausgabeSchicht =
        {
            new Knoten("v1",[versteckteSchicht[0],versteckteSchicht[1]],[r.NextDouble(),r.NextDouble()]), 
        };


        Knoten[][] netz = [eingabeSchicht, versteckteSchicht, ausgabeSchicht];

        for(int epoche = 0; epoche < 20000; epoche++)
        {
            for(int i = 0; i < möglicheEingaben.GetLength(0); i++)
            {
                eingabeSchicht[0].Ausgabe = möglicheEingaben[i,0];
                eingabeSchicht[1].Ausgabe = möglicheEingaben[i,1];

                Forward(netz);

                BackPropagation(netz,[möglicheAusgaben[i]], 0.5);

            }
        }

        Console.WriteLine("Trainiertes Netz:");

        for(int i = 0; i < möglicheEingaben.GetLength(0); i++)
        {
            eingabeSchicht[0].Ausgabe = möglicheEingaben[i,0];
            eingabeSchicht[1].Ausgabe = möglicheEingaben[i,1];

            Forward(netz);

            Console.WriteLine($"{möglicheEingaben[i,0]} XOR {möglicheEingaben[i,1]}: {ausgabeSchicht[0].Ausgabe}");
        }

      
    }
    public static void Forward(Knoten[][] schichten)
    {
        for(int i = 1; i < schichten.GetLength(0); i++)
        {
            foreach(Knoten k in schichten[i])
            {
                k.Berechne();
            }
        }
    }
    public static void BackPropagation(Knoten[][] schichten, double[] erwartungswert, double lernrate)
    {
        Knoten[] ausgabeSchicht = schichten[^1];
        for (int i = 0; i < ausgabeSchicht.Length; i++)
        {
            Knoten k = ausgabeSchicht[i];
            k.Delta = (erwartungswert[i] - k.Ausgabe) * k.Ausgabe * (1- k.Ausgabe);
        }

        for(int s = schichten.Length - 2; s >= 1; s--)
        {
            foreach(Knoten k in schichten[s])
            {
                double fehler = 0;
                foreach(Knoten nachfolger in schichten[s + 1])
                {
                    int index = Array.IndexOf(nachfolger.Eingänge, k);
                    fehler += nachfolger.Delta * nachfolger.Gewichte[index];

                }
                k.Delta = fehler * k.Ausgabe * (1 - k.Ausgabe);
            }
        }

        for(int p = 1; p < schichten.Length; p++)
        {
            foreach(Knoten k in schichten[p])
            {
                for(int q = 0; q < k.Eingänge.Length; q++)
                {
                    k.Gewichte[q] += lernrate * k.Delta * k.Eingänge[q].Ausgabe;
                }
                k.Bias += lernrate * k.Delta;
            }
        }
    }


}
public class Knoten
{
    public string Name {get; set;}
    public Knoten[] Eingänge {get;set;}
    public double[] Gewichte {get; set;}
    public double Ausgabe {get; set;}
    public double Delta {get; set;}
    public double Bias { get; set; }


    public Knoten(string name, Knoten[] eingänge = null, double[] gewichte = null)
    {
        Eingänge = eingänge ?? [];
        Gewichte = gewichte ?? [];
        Name = name;
    }

    public void Berechne()
    {
        double summe = Bias;
        for(int i = 0; i < Eingänge.Length; i++)
        {
            summe += Eingänge[i].Ausgabe * Gewichte[i];

        }
        Ausgabe = Sigmoid(summe);
    }
    
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}

