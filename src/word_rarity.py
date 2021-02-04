import numexpr
import numpy as np


def uni_leipzig_top100de():
    # source: https://web.archive.org/web/20091217142521/http://wortschatz.uni-leipzig.de/Papers/top100de.txt
    # returns 91 unique german lower case words
    words = (
        "der die und in den von zu das mit sich des auf für ist im dem nicht ein "
        + "Die eine als auch es an werden aus er hat dass sie nach wird bei einer "
        + "Der um am sind noch wie einem über einen Das so Sie zum war haben nur oder "
        + "aber vor zur bis mehr durch man sein wurde sei In Prozent hatte kann gegen "
        + "vom können schon wenn habe seine Mark ihre dann unter wir soll ich eines "
        + "Es Jahr zwei Jahren diese dieser wieder keine Uhr seiner worden Und will "
        + "zwischen Im immer Millionen Ein was sagte"
    )
    words = words.lower()
    words = words.split()
    words = np.array(words)
    words = np.unique(words)
    return words


def uni_leipzig_top1000de():
    # source: https://web.archive.org/web/20090919061930/http://wortschatz.uni-leipzig.de/Papers/top1000de.txt
    # returns 901 unique german lower case words
    words = (
        "der die und in den von zu das mit sich des auf für ist im dem nicht ein Die "
        + "eine als auch es an werden aus er hat dass sie nach wird bei einer Der um "
        + "am sind noch wie einem über einen Das so Sie zum war haben nur oder aber "
        + "vor zur bis mehr durch man sein wurde sei In Prozent hatte kann gegen vom "
        + "können schon wenn habe seine Mark ihre dann unter wir soll ich eines Es "
        + "Jahr zwei Jahren diese dieser wieder keine Uhr seiner worden Und will "
        + "zwischen Im immer Millionen Ein was sagte Er gibt alle DM diesem seit "
        + "muss wurden beim doch jetzt waren drei Jahre Mit neue neuen damit bereits "
        + "da Auch ihr seinen müssen ab ihrer Nach ohne sondern selbst ersten nun "
        + "etwa Bei heute ihren weil ihm seien Menschen Deutschland anderen werde "
        + "Ich sagt Wir Eine rund Für Aber ihn Ende jedoch Zeit sollen ins Wenn So "
        + "seinem uns Stadt geht Doch sehr hier ganz erst wollen Berlin vor allem "
        + "sowie hatten kein deutschen machen lassen Als Unternehmen andere ob "
        + "dieses steht dabei wegen weiter denn beiden einmal etwas Wie nichts "
        + "allerdings vier gut viele wo viel dort alles Auf wäre SPD kommt "
        + "vergangenen denen fast fünf könnte nicht nur hätten Frau Am dafür kommen"
        + " diesen letzten zwar Diese großen dazu Von Mann Da sollte würde also "
        + "bisher Leben Milliarden Welt Regierung konnte ihrem Frauen während Land "
        + "zehn würden stehen ja USA heißt dies zurück Kinder dessen ihnen deren "
        + "sogar Frage gewesen erste gab liegt gar davon gestern geben Teil Polizei "
        + "dass hätte eigenen kaum sieht große Denn weitere Was sehen macht "
        + "Angaben weniger gerade lässt Geld München deutsche allen darauf wohl "
        + "später könne deshalb aller kam Arbeit mich gegenüber nächsten bleibt "
        + "wenig lange gemacht Wer Dies Fall mir gehen Berliner mal Weg CDU wollte "
        + "sechs keinen Woche dagegen alten möglich gilt erklärte müsse Dabei "
        + "könnten Geschichte zusammen finden Tag Art erhalten Man Dollar Wochen "
        + "jeder nie bleiben besonders Jahres Deutschen Den Zu zunächst derzeit "
        + "allein deutlich Entwicklung weiß einige sollten Präsident geworden statt "
        + "Bonn Platz inzwischen Nur Freitag Um pro seines Damit Montag Europa "
        + "schließlich Sonntag einfach gehört eher oft Zahl neben hält weit Partei "
        + "meisten Thema zeigt Politik Aus zweiten Januar insgesamt je musste Anfang "
        + "hinter ebenfalls ging Mitarbeiter darüber vielen Ziel darf Seite fest hin "
        + "erklärt Namen Haus An Frankfurt Gesellschaft Mittwoch damals Dienstag "
        + "Hilfe Mai Markt Seit Tage Donnerstag halten gleich nehmen solche "
        + "Entscheidung besser alte Leute Ergebnis Samstag Dass sagen System März tun "
        + "Monaten kleinen lang Nicht knapp bringen wissen Kosten Erfolg bekannt "
        + "findet daran künftig wer acht Grünen schnell Grund scheint Zukunft "
        + "Stuttgart bin liegen politischen Gruppe Rolle stellt Juni sieben September "
        + "nämlich Männer Oktober Mrd überhaupt eigene Dann gegeben Außerdem "
        + "Stunden eigentlich Meter ließ Probleme vielleicht ebenso Bereich zum "
        + "Beispiel Bis Höhe Familie Während Bild Ländern Informationen Frankreich "
        + "Tagen schwer zuvor Vor genau April stellen neu erwartet Hamburg sicher "
        + "führen Mal über mehrere Wirtschaft Mio Programm offenbar Hier weiteren "
        + "natürlich konnten stark Dezember Juli ganze kommenden Kunden bekommen "
        + "eben kleine trotz wirklich Lage Länder leicht gekommen Spiel laut "
        + "November kurz politische fährt innerhalb unsere meint immer wieder Form "
        + "Münchner AG anders ihres völlig beispielsweise gute bislang August Hand "
        + "jede GmbH Film Minuten erreicht beide Musik Kritik Mitte Verfügung Buch "
        + "dürfen Unter jeweils einigen Zum Umsatz spielen Daten welche müssten hieß "
        + "paar nachdem Kunst Euro gebracht Problem Noch jeden Ihre Sprecher recht "
        + "erneut längst europäischen Sein Eltern Beginn besteht Seine mindestens "
        + "machte Jetzt bietet außerdem Bürger Trainer bald Deutsche Schon Fragen "
        + "klar Durch Seiten gehören Dort erstmals Februar zeigen Titel Stück "
        + "größten FDP setzt Wert Frankfurter Staat möchte daher wolle "
        + "Bundesregierung lediglich Nacht Krieg Opfer Tod nimmt Firma zuletzt Werk "
        + "hohen leben unter anderem Dieser Kirche weiterhin gebe gestellt "
        + "Mitglieder Rahmen zweite Paris Situation gefunden Wochenende "
        + "internationalen Wasser Recht sonst stand Hälfte Möglichkeit versucht "
        + "blieb junge Mehrheit Straße Sache arbeiten Monate Mutter berichtet letzte "
        + "Gericht wollten Ihr zwölf zumindest Wahl genug Weise Vater Bericht "
        + "amerikanischen hoch beginnt Wort obwohl Kopf spielt Interesse Westen "
        + "verloren Preis Erst jedem erreichen setzen spricht früher teilte Landes "
        + "zudem einzelnen bereit Blick Druck Bayern Kilometer gemeinsam Bedeutung "
        + "Chance Politiker Dazu Zwei besten Ansicht endlich Stelle direkt Beim "
        + "Bevölkerung Viele solchen Alle solle jungen Einsatz richtig größte "
        + "sofort neuer ehemaligen unserer dürfte schaffen Augen Russland Internet "
        + "Allerdings Raum Mannschaft neun kamen Ausstellung Zeiten Dem einzige "
        + "meine Nun Verfahren Angebot Richtung Projekt niemand Kampf weder "
        + "tatsächlich Personen dpa Heute geführt Gespräch Kreis Hamburger Schule "
        + "guten Hauptstadt durchaus Zusammenarbeit darin Amt Schritt meist groß "
        + "zufolge Sprache Region Punkte Vergleich genommen gleichen du Ob Soldaten "
        + "Universität verschiedenen Kollegen neues Bürgermeister Angst stellte "
        + "Sommer danach anderer gesagt Sicherheit Macht Bau handelt Folge Bilder "
        + "lag Osten Handel sprach Aufgabe Chef frei dennoch DDR hohe Firmen bzw "
        + "Koalition Mädchen Zur entwickelt fand Diskussion bringt Deshalb Hause "
        + "Gefahr per zugleich früheren dadurch ganzen abend erzählt Streit "
        + "Vergangenheit Parteien Verhandlungen jedenfalls gesehen französischen "
        + "Trotz darunter Spieler forderte Beispiel Meinung wenigen Publikum sowohl "
        + "meinte mag Auto Lösung Boden Einen Präsidenten hinaus Zwar verletzt "
        + "zB weltweit Sohn bevor Peter mussten keiner Produktion Ort braucht "
        + "Zusammenhang Kind Verein sprechen Aktien gleichzeitig London sogenannten "
        + "Richter geplant Italien Mittel her freilich Mensch großer Bonner wenige "
        + "öffentlichen Unterstützung dritten nahm Bundesrepublik Arbeitsplätze "
        + "bedeutet Feld Dr. Bank oben gesetzt Ausland Ministerpräsident Vertreter "
        + "jedes ziehen Parlament berichtete Dieses China aufgrund Stellen "
        + "warum Kindern heraus heutigen Anteil Herr öffentlichkeit Abend Selbst "
        + "Liebe Neben rechnen fällt New York Industrie WELT Stuttgarter wären "
        + "Vorjahr Sicht Idee Banken verlassen Leiter Bühne insbesondere offen "
        + "stets Theater ändern entschieden Staaten Experten Gesetz Geschäft Tochter "
        + "angesichts gelten Mehr erwarten läuft fordert Japan Sieg Ist Stimmen "
        + "wählen russischen gewinnen CSU bieten Nähe jährlich Bremen Schüler "
        + "Rede Funktion Zuschauer hingegen anderes Führung Besucher Drittel Moskau "
        + "immerhin Vorsitzende Urteil Schließlich Kultur betonte mittlerweile "
        + "Saison Konzept suchen Zahlen Roman Gewalt Köln gesamte indem EU Stunde "
        + "ehemalige Auftrag entscheiden genannt tragen Börse langen häufig Chancen "
        + "Vor allem Position alt Luft Studenten übernehmen stärker ohnehin zeigte "
        + "geplanten Reihe darum verhindern begann Medien verkauft Minister wichtig "
        + "amerikanische sah gesamten einst verwendet vorbei Behörden helfen Folgen "
        + "bezeichnet"
    )
    words = words.lower()
    words = words.split()
    words = np.array(words)
    words = np.unique(words)
    return words


if __name__ == "__main__":
    numexpr.print_versions()

    top100 = uni_leipzig_top100de()
    top1000 = uni_leipzig_top1000de()
    print(top100.size)
    print(top1000.size)
